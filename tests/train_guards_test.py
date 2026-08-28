"""
Guards on the in-batch / cross-batch training entry points.

Tevatron's driver resumes from whatever `get_last_checkpoint(output_dir)` finds and
`--overwrite_output_dir` does not suppress it, so a re-run into a finished directory
used to resume, take zero optimizer steps, re-save the old weights and exit 0. These
tests pin the gate that prevents it, the success validation that would have caught
it, the fixed-probe ranking signal, and the cross-batch pool contract.

Run: python tests/train_guards_test.py
"""
import json
import math
import os
import sys
import shutil
import tempfile
import time
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

import torch                                                      # noqa: E402
from utils.helpers import (                                       # noqa: E402
    RUN_MANIFEST_NAME, TRAINING_LOG_NAME, RunDirectoryError, append_jsonl,
    assert_training_succeeded, attach_training_diagnostics, build_run_manifest,
    prepare_output_dir,
    probe_triples_from_mixture, ranking_probe, require_recipe_keys,
)

sys.path.insert(0, str(project_root / 'tests'))
from fast_grass_test import DEVICE, GradMockModel, MockTokenizer   # noqa: E402


# ---- fixtures --------------------------------------------------------------

def _assert_raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        assert contains is None or contains in str(e), str(e)
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


def _mixture(tmp, n=5, name="train_hq.jsonl"):
    """A mixture file in the shape require_mixture_files hands to the manifest."""
    path = Path(tmp) / name
    with open(path, 'w') as f:
        for i in range(n):
            f.write(json.dumps({
                'query_id': str(i), 'query': f'q{i}',
                'positive_passages': [{'docid': f'p{i}', 'text': f'pos {i}'}],
                'negative_passages': [{'docid': f'n{i}', 'text': f'neg {i}'}],
            }) + '\n')
    return path


_RECIPE = {'model_name': 'm', 'batch_size': 4, 'train_group_size': 2,
           'learning_rate': 1e-5, 'num_epochs': 2}
_CTX = {'args': _RECIPE, 'base_model': '/nonexistent/base'}


def _manifest(tmp, **over):
    recipe = {**_RECIPE, **over.pop('recipe', {})}
    kwargs = {'data_files': [_mixture(tmp)], 'world_size': 1,
              'negative_pool_size': 7, 'optimizer_steps': 10}
    kwargs.update(over)
    return build_run_manifest('inbatch', {**_CTX, 'args': recipe}, recipe, **kwargs)


def _ckpt(out, step=17202, valid=True, state=True):
    d = Path(out) / f"checkpoint-{step}"
    d.mkdir(parents=True, exist_ok=True)
    if valid:
        (d / "optimizer.pt").write_bytes(b"x")
    if state:
        # HF writes this into every checkpoint; --resume reads its global_step as
        # the invocation baseline, so a fixture without it is not a real checkpoint.
        (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return d


def _safetensors(path, tensors=1):
    """A minimal well-formed safetensors file: 8-byte length + JSON header."""
    header = {f"t{i}": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
              for i in range(tensors)}
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, 'little') + raw + b"\x00" * 4)


def _trained_output(tmp, manifest, *, steps=100, loss=0.5, tensors=1):
    """An output dir that looks like a run which really did train."""
    out = Path(tmp) / "out"
    prepare_output_dir(out, manifest)
    # A zero-step run logs no loss at all: on_log never fires. Only the probes,
    # which sit at global_step 0, reach the file.
    for step in ([1, steps] if steps else []):
        append_jsonl(out / TRAINING_LOG_NAME,
                     {"global_step": step, "loss": loss, "learning_rate": 1e-5,
                      "grad_norm": 1.2})
    append_jsonl(out / TRAINING_LOG_NAME,
                 {"global_step": 0, "phase": "begin", "rank_acc": 0.5, "margin_mean": 0.0})
    append_jsonl(out / TRAINING_LOG_NAME,
                 {"global_step": steps, "phase": "end", "rank_acc": 0.9, "margin_mean": 0.3})
    (out / "config.json").write_text(json.dumps({"model_type": "xlm-roberta"}))
    _safetensors(out / "model.safetensors", tensors)
    return out


# ---- output directory gate -------------------------------------------------

def test_stale_checkpoints_removed_on_fresh_run():
    """The whole point: get_last_checkpoint must find nothing, or Tevatron resumes."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = Path(tmp) / "out"
        prepare_output_dir(out, m)          # establishes the manifest
        _ckpt(out)
        prepare_output_dir(out, _manifest(tmp))
        assert list(out.glob("checkpoint-*")) == [], list(out.glob("checkpoint-*"))
        assert (out / RUN_MANIFEST_NAME).is_file()


def test_unidentifiable_checkpoints_are_refused_not_deleted():
    """Checkpoints with no manifest predate the gate: nothing says whose they are.

    Deleting them silently is how a previous run's artifacts get discarded by someone
    who only meant to start a new run. It happened once; this pins the refusal.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        _ckpt(out, 500)
        _ckpt(out, 1000)
        _assert_raises(RunDirectoryError, lambda: prepare_output_dir(out, _manifest(tmp)),
                       "2 checkpoint(s) but no run_manifest.json")
        assert len(list(out.glob("checkpoint-*"))) == 2, "refusal must not delete"
        # --overwrite makes the discard explicit, and only then do they go.
        prepare_output_dir(out, _manifest(tmp), overwrite=True)
        assert list(out.glob("checkpoint-*")) == []


def test_unfinished_run_does_not_block_a_corrected_rerun():
    """A job that dies at startup must not lock the directory against its own fix.

    The manifest is written BEFORE training. Job 14948 failed at model load with an
    unresolvable base_model; once the cache was seeded the fingerprint changed and the
    dead run's manifest refused the retry.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        prepare_output_dir(out, _manifest(tmp))            # run 1 starts...
        assert not json.loads((out / RUN_MANIFEST_NAME).read_text()).get('finished_at')
        # ...and dies. Run 2 has a different base_model, and must be allowed.
        second = _manifest(tmp, recipe={'learning_rate': 9e-9})
        prepare_output_dir(out, second)
        assert json.loads((out / RUN_MANIFEST_NAME).read_text())['fingerprint'] == second['fingerprint']


def test_finished_run_still_blocks_a_different_config():
    """The protection must survive: a COMPLETED run is what we refuse to clobber."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _trained_output(tmp, m, steps=50)
        assert_training_succeeded(out, m)                  # stamps finished_at
        assert json.loads((out / RUN_MANIFEST_NAME).read_text())['finished_at']
        _assert_raises(RunDirectoryError,
                       lambda: prepare_output_dir(out, _manifest(tmp, recipe={'batch_size': 8})),
                       "different configuration")


def test_clean_dir_with_no_manifest_still_proceeds():
    """A genuinely fresh directory must not need --overwrite."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        prepare_output_dir(out, _manifest(tmp))
        assert (out / RUN_MANIFEST_NAME).is_file()


def test_fresh_run_over_same_fingerprint_is_allowed():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = Path(tmp) / "out"
        prepare_output_dir(out, m)
        prepare_output_dir(out, _manifest(tmp))          # same inputs -> same fingerprint
        assert (out / RUN_MANIFEST_NAME).is_file()


def test_different_fingerprint_refuses():
    """Only a FINISHED run blocks: an unfinished one has nothing to protect."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _trained_output(tmp, m, steps=50)
        assert_training_succeeded(out, m)               # stamps finished_at
        msg = _assert_raises(RunDirectoryError,
                             lambda: prepare_output_dir(out, _manifest(tmp, recipe={'batch_size': 8})),
                             "different configuration")
        assert "effective_config" in msg, msg


def test_overwrite_permits_an_incompatible_dir():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        prepare_output_dir(out, _manifest(tmp))
        other = _manifest(tmp, recipe={'batch_size': 8})
        prepare_output_dir(out, other, overwrite=True)
        assert json.loads((out / RUN_MANIFEST_NAME).read_text())['fingerprint'] == other['fingerprint']


def test_resume_without_a_prior_manifest_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        _assert_raises(RunDirectoryError,
                       lambda: prepare_output_dir(Path(tmp) / "out", _manifest(tmp), resume=True),
                       "nothing to resume from")


def test_resume_with_an_incompatible_manifest_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        prepare_output_dir(out, _manifest(tmp))
        _ckpt(out)
        _assert_raises(RunDirectoryError,
                       lambda: prepare_output_dir(out, _manifest(tmp, recipe={'num_epochs': 9}),
                                                  resume=True),
                       "different configuration")


def test_resume_keeps_checkpoints():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        m = _manifest(tmp)
        prepare_output_dir(out, m)
        _ckpt(out, 500)
        prepare_output_dir(out, _manifest(tmp), resume=True)
        assert [p.name for p in out.glob("checkpoint-*")] == ["checkpoint-500"]


def test_resume_without_a_checkpoint_refuses():
    """A matching manifest with no checkpoint would silently start from scratch."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        prepare_output_dir(out, _manifest(tmp))
        _assert_raises(RunDirectoryError,
                       lambda: prepare_output_dir(out, _manifest(tmp), resume=True),
                       "no checkpoint-*")


def test_unreadable_manifest_is_incompatible_not_absent():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        out.mkdir()
        (out / RUN_MANIFEST_NAME).write_text("{not json")
        _assert_raises(RunDirectoryError, lambda: prepare_output_dir(out, _manifest(tmp)))


# ---- manifest --------------------------------------------------------------

def test_fingerprint_is_stable_across_two_builds():
    with tempfile.TemporaryDirectory() as tmp:
        assert _manifest(tmp)['fingerprint'] == _manifest(tmp)['fingerprint']


def test_fingerprint_moves_with_config_data_and_world_size():
    with tempfile.TemporaryDirectory() as tmp:
        base = _manifest(tmp)['fingerprint']
        assert _manifest(tmp, recipe={'learning_rate': 2e-5})['fingerprint'] != base
        assert _manifest(tmp, world_size=2)['fingerprint'] != base
        changed = _mixture(tmp, n=9)
        assert build_run_manifest('inbatch', _CTX, _RECIPE, data_files=[changed],
                                  world_size=1, negative_pool_size=7,
                                  optimizer_steps=10)['fingerprint'] != base


def test_manifest_records_provenance_without_raising():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        assert m['data_files'][0]['lines'] == 5, m['data_files']
        assert len(m['data_files'][0]['sha256']) == 64
        assert set(m['code_revision']) == {'git_sha', 'git_dirty'}
        assert 'torch' in m['dependencies'] and 'pyserini' in m['dependencies']
        # Distribution name, not import name: a 'grad-cache' lookup silently records
        # null for a GradCache that is in fact installed.
        assert m['dependencies']['GradCache'] is not None, \
            "GradCache must resolve by its distribution name"
        # An absent package records None rather than exploding the run.
        assert m['dependencies']['faiss-gpu'] is None or isinstance(
            m['dependencies']['faiss-gpu'], str)
        assert m['base_model_exists'] is False
        assert m['negative_pool_size'] == 7 and m['optimizer_steps_planned'] == 10


# ---- success validation ----------------------------------------------------

def test_zero_optimizer_steps_fails():
    """The exact silent-success path: resumed, trained nothing, re-saved."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _trained_output(tmp, m, steps=0)
        _assert_raises(RunDirectoryError, lambda: assert_training_succeeded(out, m),
                       "no new optimizer steps")


def test_non_finite_loss_fails():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _trained_output(tmp, m, loss=float('nan'))
        _assert_raises(RunDirectoryError, lambda: assert_training_succeeded(out, m),
                       "non-finite loss")


def test_missing_diagnostics_fails():
    """No log AND no trainer_state.json to fall back on: still no evidence of training.

    The message moved when the trainer_state.json fallback was added -- a missing log
    is now one way of reaching the zero-step branch rather than its own early return.
    """
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _trained_output(tmp, m)
        (out / TRAINING_LOG_NAME).unlink()
        _assert_raises(RunDirectoryError, lambda: assert_training_succeeded(out, m),
                       "no new optimizer steps")


def test_stale_checkpoint_fails():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _trained_output(tmp, m)
        old = m['started_at_epoch'] - 3600
        os.utime(out / "model.safetensors", (old, old))
        _assert_raises(RunDirectoryError, lambda: assert_training_succeeded(out, m),
                       "predates the start of this run")


def test_truncated_checkpoint_fails():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _trained_output(tmp, m)
        raw = (out / "model.safetensors").read_bytes()
        (out / "model.safetensors").write_bytes(raw[:12])          # header cut short
        _assert_raises(Exception, lambda: assert_training_succeeded(out, m), "truncated")


def test_success_appends_to_the_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _trained_output(tmp, m, steps=250)
        stored = assert_training_succeeded(out, m)
        assert stored['final_global_step'] == 250, stored['final_global_step']
        assert len(stored['probe']) == 2, stored['probe']
        assert stored['fingerprint'] == m['fingerprint']
        on_disk = json.loads((out / RUN_MANIFEST_NAME).read_text())
        assert on_disk['final_global_step'] == 250


# ---- fixed probe -----------------------------------------------------------

class _RankModel(torch.nn.Module):
    """Emits a fixed embedding per text so margins are exactly predictable."""

    def __init__(self, table):
        super().__init__()
        self.table = table
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, **inputs):
        vecs = torch.stack([self.table[int(i)] for i in inputs['input_ids'][:, 0]])
        return type('O', (), {'last_hidden_state': vecs.unsqueeze(1)})()


class _IdTokenizer:
    """Maps each distinct text to the row index the model should emit."""

    def __init__(self, order):
        self.order = {t: i for i, t in enumerate(order)}

    def __call__(self, batch, **kw):
        ids = torch.tensor([[self.order[t]] for t in batch])
        return type('B', (), {'to': lambda _s, _d: {'input_ids': ids,
                                                    'attention_mask': torch.ones_like(ids)}})()


def _probe_setup():
    triples = [("q0", "p0", "n0"), ("q1", "p1", "n1")]
    texts = ["q0", "q1", "p0", "p1", "n0", "n1"]
    table = torch.tensor([
        [1.0, 0.0], [1.0, 0.0],          # queries
        [1.0, 0.0], [0.0, 1.0],          # positives: q0 aligned, q1 orthogonal
        [0.0, 1.0], [1.0, 0.0],          # negatives: q0 orthogonal, q1 aligned
    ])
    return triples, _RankModel(table), _IdTokenizer(texts)


def test_probe_margin_and_rank_acc_are_exact():
    triples, model, tok = _probe_setup()
    out = ranking_probe(model, tok, triples, torch.device('cpu'), 8, 8)
    # q0: 1 - 0 = +1 (correct);  q1: 0 - 1 = -1 (wrong)  ->  mean 0, acc 0.5
    assert out['n'] == 2 and abs(out['margin_mean']) < 1e-6, out
    assert abs(out['rank_acc'] - 0.5) < 1e-6, out


def test_probe_refuses_a_none_tokenizer_with_a_clear_message():
    """Tevatron builds Trainer WITHOUT tokenizer=, so the callback kwarg is always None.

    This shipped as a bare `TypeError: 'NoneType' object is not callable` from inside
    encode_batch_tensor, recorded in training_log.jsonl on a live 12-hour run.
    """
    triples, model, _tok = _probe_setup()
    _assert_raises(ValueError,
                   lambda: ranking_probe(model, None, triples, torch.device('cpu'), 8, 8),
                   "tokenizer=None")


def test_entry_points_supply_their_own_probe_tokenizer():
    """Both Tevatron trainers must not rely on the callback's tokenizer kwarg."""
    for name in ("train_inbatch.py", "train_crossbatch.py"):
        src = (project_root / "scripts" / name).read_text()
        assert "AutoTokenizer.from_pretrained(ctx['base_model'])" in src, name
        assert "tokenizer or probe_tokenizer" in src, name


def test_probe_is_deterministic():
    triples, model, tok = _probe_setup()
    a = ranking_probe(model, tok, triples, torch.device('cpu'), 8, 8)
    b = ranking_probe(model, tok, triples, torch.device('cpu'), 8, 8)
    assert a == b, (a, b)


def test_probe_restores_train_mode():
    """A probe that leaves dropout off silently changes the next training step."""
    triples, model, tok = _probe_setup()
    model.train()
    ranking_probe(model, tok, triples, torch.device('cpu'), 8, 8)
    assert model.training is True


def test_probe_triples_are_the_last_records_of_train_hq():
    with tempfile.TemporaryDirectory() as tmp:
        hq = _mixture(tmp, n=10)
        other = _mixture(tmp, n=3, name="train_vl.jsonl")
        triples = probe_triples_from_mixture([other, hq], n=4)
        assert [t[0] for t in triples] == ['q6', 'q7', 'q8', 'q9'], triples
        assert probe_triples_from_mixture([other, hq], n=4) == triples   # deterministic


def test_probe_triples_skip_records_without_a_negative():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train_hq.jsonl"
        path.write_text(
            json.dumps({'query': 'good', 'positive_passages': [{'text': 'p'}],
                        'negative_passages': [{'text': 'n'}]}) + '\n' +
            json.dumps({'query': 'bad', 'positive_passages': [{'text': 'p'}],
                        'negative_passages': []}) + '\n')
        assert [t[0] for t in probe_triples_from_mixture([path], n=4)] == ['good']


# ---- callback against a real HF Trainer ------------------------------------

class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(4, 1)

    def forward(self, x=None, labels=None):
        out = self.l(x).squeeze(-1)
        return {"loss": ((out - labels) ** 2).mean(), "logits": out}


class _TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return {"x": torch.randn(4, generator=g), "labels": torch.tensor(1.0)}


def _drive_trainer(tmp, probe_fn, **arg_over):
    """Run a real transformers Trainer so the callback is exercised as it will be.

    The Tevatron entry points hand Trainer an argv list and never see the object,
    so DEFAULT_CALLBACKS registration is the only attachment point -- and the only
    way to know it still works is to build a Trainer.
    """
    from transformers import Trainer, TrainingArguments
    import transformers.trainer as _tm
    log = attach_training_diagnostics(tmp, probe_fn)
    try:
        args = TrainingArguments(output_dir=tmp, per_device_train_batch_size=8,
                                 num_train_epochs=2, logging_steps=1, report_to=[],
                                 save_strategy="no", use_cpu=True,
                                 disable_tqdm=True, **arg_over)
        Trainer(model=_TinyModel(), args=args, train_dataset=_TinyDataset()).train()
    finally:
        _tm.DEFAULT_CALLBACKS[:] = [
            cb for cb in _tm.DEFAULT_CALLBACKS
            if getattr(cb, '__name__', '') != 'TrainingDiagnosticsCallback']
    return [json.loads(line) for line in Path(log).read_text().splitlines() if line.strip()]


def test_callback_persists_loss_lr_and_pre_clipping_grad_norm():
    with tempfile.TemporaryDirectory() as tmp:
        records = _drive_trainer(tmp, lambda m, t: {"rank_acc": 0.7, "margin_mean": 0.1})
        steps = [r for r in records if r.get('phase') is None]
        assert len(steps) == 8, len(steps)
        for key in ("loss", "learning_rate", "grad_norm"):
            assert all(r.get(key) is not None for r in steps), (key, steps[:2])
        assert [r['global_step'] for r in steps] == list(range(1, 9))


def test_callback_probes_at_three_points():
    """>= 2 task-relevant points is the requirement; begin/50%/end gives three."""
    with tempfile.TemporaryDirectory() as tmp:
        records = _drive_trainer(tmp, lambda m, t: {"rank_acc": 0.7, "margin_mean": 0.1})
        phases = [r['phase'] for r in records if r.get('phase')]
        assert phases == ['begin', 'step_50pct', 'end'], phases


def test_callback_records_a_failing_probe_instead_of_killing_the_run():
    def boom(model, tokenizer):
        raise RuntimeError("probe exploded")

    with tempfile.TemporaryDirectory() as tmp:
        records = _drive_trainer(tmp, boom)
        probes = [r for r in records if r.get('phase')]
        assert probes and all('probe exploded' in r.get('error', '') for r in probes), probes
        assert len([r for r in records if r.get('phase') is None]) == 8


def test_callback_registration_is_idempotent():
    import transformers.trainer as _tm
    with tempfile.TemporaryDirectory() as tmp:
        before = len(_tm.DEFAULT_CALLBACKS)
        try:
            attach_training_diagnostics(tmp)
            attach_training_diagnostics(tmp)
            names = [getattr(cb, '__name__', '') for cb in _tm.DEFAULT_CALLBACKS]
            assert names.count('TrainingDiagnosticsCallback') == 1, names
            assert len(_tm.DEFAULT_CALLBACKS) == before + 1
        finally:
            _tm.DEFAULT_CALLBACKS[:] = [
                cb for cb in _tm.DEFAULT_CALLBACKS
                if getattr(cb, '__name__', '') != 'TrainingDiagnosticsCallback']


# ---- cross-batch pool contract ---------------------------------------------

def _cb_recipe(**over):
    base = {'per_device_batch_size': 512, 'target_batch_size': 1024,
            'gradient_accumulation_steps': 1, 'train_group_size': 2, 'grad_cache': True}
    base.update(over)
    return base


def test_crossbatch_uses_non_reentrant_checkpointing():
    """GradCache + reentrant checkpointing + DDP = "marked as ready twice".

    Job 15001 died in grad_cache's second backward; job 15025 died identically because
    the first fix used setdefault on the OUTER kwargs, and Trainer passes
    gradient_checkpointing_kwargs={} -- an empty dict, not None -- so the key already
    existed and the default never applied. This test drives the patch the way Trainer
    actually calls it instead of grepping the source.
    """
    import importlib
    cb = importlib.import_module("train_crossbatch")
    seen = {}

    class _Encoder:
        def gradient_checkpointing_enable(self, **kw):
            seen.update(kw)

    class _Model:
        encoder = _Encoder()

    # exactly transformers/trainer.py:1985-1990
    cb._tevatron_gc_enable(_Model(), gradient_checkpointing_kwargs={})
    assert seen.get("gradient_checkpointing_kwargs", {}).get("use_reentrant") is False, seen

    # and an explicit caller preference must win over our default
    seen.clear()
    cb._tevatron_gc_enable(_Model(), gradient_checkpointing_kwargs={"use_reentrant": True})
    assert seen["gradient_checkpointing_kwargs"]["use_reentrant"] is True, seen

    # in-batch is single-process and must keep the plain no-arg call
    import ast
    ib = (project_root / "scripts" / "train_inbatch.py").read_text()
    ibfn = [n for n in ast.walk(ast.parse(ib))
            if isinstance(n, ast.FunctionDef) and n.name == "_tevatron_gc_enable"][0]
    assert "use_reentrant" not in ast.unparse(ibfn), "in-batch must keep the no-arg call"


def test_pool_world_size_two_gives_1024_2048_2047():
    from train_crossbatch import check_batch_invariants
    pool = check_batch_invariants(_cb_recipe(), world_size=2)
    assert pool == {'world_size': 2, 'queries': 1024, 'passages': 2048,
                    'negatives_per_query': 2047}, pool


def test_pool_single_process_refuses():
    """is_ddp is False without torchrun; the all-gather vanishes silently."""
    from train_crossbatch import check_batch_invariants
    _assert_raises(ValueError, lambda: check_batch_invariants(_cb_recipe(), world_size=1),
                   "torchrun --nproc_per_node=2")


def test_pool_accumulation_refuses():
    from train_crossbatch import check_batch_invariants
    _assert_raises(ValueError,
                   lambda: check_batch_invariants(_cb_recipe(gradient_accumulation_steps=2),
                                                  world_size=2),
                   "does NOT enlarge")


def test_pool_grad_cache_off_refuses():
    from train_crossbatch import check_batch_invariants
    _assert_raises(ValueError,
                   lambda: check_batch_invariants(_cb_recipe(grad_cache=False), world_size=2),
                   "grad_cache is false")


# ---- config ownership ------------------------------------------------------

def test_config_unused_key_refuses():
    _assert_raises(ValueError,
                   lambda: require_recipe_keys("crossbatch", {'a': 1, 'ghost': 2}, ('a',)),
                   "declared but never consumed: ghost")


def test_config_missing_key_refuses():
    _assert_raises(ValueError,
                   lambda: require_recipe_keys("inbatch", {'a': 1}, ('a', 'b')),
                   "consumed but not declared: b")


def test_config_optional_keys_may_be_absent_or_present():
    require_recipe_keys("crossbatch", {'a': 1}, ('a',), optional=('lora',))
    require_recipe_keys("crossbatch", {'a': 1, 'lora': True}, ('a',), optional=('lora',))


def test_real_recipes_agree_with_the_entry_points():
    """The live config must satisfy the very checks the trainers run at startup."""
    import train_inbatch
    import train_crossbatch
    from utils.helpers import load_config
    training = load_config()['training']
    require_recipe_keys("inbatch", training['inbatch'], train_inbatch.CONSUMED_KEYS)
    require_recipe_keys("crossbatch", training['crossbatch'],
                        train_crossbatch.CONSUMED_KEYS, train_crossbatch.OPTIONAL_KEYS)
    from train_crossbatch import check_batch_invariants
    assert check_batch_invariants(training['crossbatch'], 2)['negatives_per_query'] == 2047


# ---- diagnostics must never kill a run (job 14990) --------------------------

def _failing_open(fail_times, real=None):
    """A builtins.open that raises EREMOTEIO for the first `fail_times` calls."""
    real = real or open
    state = {'n': 0}

    def fake(path, *a, **kw):
        if str(path).endswith(TRAINING_LOG_NAME) and state['n'] < fail_times:
            state['n'] += 1
            raise OSError(121, "Remote I/O error")
        return real(path, *a, **kw)
    return fake, state


def test_append_jsonl_survives_a_failing_open():
    """Job 14990 died at step 3000/10314 because this raised into Trainer.log."""
    import builtins
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / TRAINING_LOG_NAME
        fake, _ = _failing_open(99)
        real_open, real_sleep = builtins.open, time.sleep
        builtins.open, time.sleep = fake, lambda *_: None
        try:
            landed = append_jsonl(log, {"global_step": 1, "loss": 0.5})
        finally:
            builtins.open, time.sleep = real_open, real_sleep
        assert landed is False, landed
        assert not log.exists(), "nothing should have been written"


def test_append_jsonl_retries_then_succeeds():
    import builtins
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / TRAINING_LOG_NAME
        fake, state = _failing_open(2)
        real_open, real_sleep = builtins.open, time.sleep
        builtins.open, time.sleep = fake, lambda *_: None
        try:
            landed = append_jsonl(log, {"global_step": 7, "loss": 0.25})
        finally:
            builtins.open, time.sleep = real_open, real_sleep
        assert landed is True, landed
        assert state['n'] == 2, state
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        assert len(lines) == 1, lines
        assert json.loads(lines[0])['global_step'] == 7


def test_fresh_run_clears_a_stale_training_log():
    """Appends accumulate, so a stale log makes max(global_step) report the OLD run."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = Path(tmp) / "out"
        prepare_output_dir(out, m)
        append_jsonl(out / TRAINING_LOG_NAME, {"global_step": 3000, "loss": 0.4})
        prepare_output_dir(out, _manifest(tmp))
        assert not (out / TRAINING_LOG_NAME).exists(), \
            (out / TRAINING_LOG_NAME).read_text()


def test_resume_keeps_the_training_log():
    """On resume the earlier records belong to the same run and must survive."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = Path(tmp) / "out"
        prepare_output_dir(out, m)
        append_jsonl(out / TRAINING_LOG_NAME, {"global_step": 2062, "loss": 0.4})
        _ckpt(out, step=2062)
        prepare_output_dir(out, _manifest(tmp), resume=True)
        assert (out / TRAINING_LOG_NAME).is_file()
        assert json.loads((out / TRAINING_LOG_NAME).read_text().splitlines()[0]
                          )['global_step'] == 2062


def test_non_finite_grad_norm_warns_but_loss_still_raises():
    """grad_norm is PRE-clipping: max_grad_norm absorbs a transient inf."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _trained_output(tmp, m)
        append_jsonl(out / TRAINING_LOG_NAME,
                     {"global_step": 101, "loss": 0.5, "grad_norm": float('inf')})
        result = assert_training_succeeded(out, m)      # must NOT raise
        assert result['final_global_step'] >= 100, result

    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _trained_output(tmp, m)
        append_jsonl(out / TRAINING_LOG_NAME,
                     {"global_step": 102, "loss": float('nan'), "grad_norm": 1.0})
        _assert_raises(RunDirectoryError,
                       lambda: assert_training_succeeded(out, m), "non-finite loss")


def test_truncated_last_log_line_tolerated():
    """Appends are not atomic; a SIGKILL can cut one mid-write."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _trained_output(tmp, m)
        with open(out / TRAINING_LOG_NAME, 'a') as f:
            f.write('{"global_step": 103, "lo')
        result = assert_training_succeeded(out, m)
        assert result['final_global_step'] >= 100, result


def test_lost_step_records_fall_back_to_trainer_state():
    """HF writes trainer_state.json itself, so dropped step writes must not condemn.

    The fallback covers the STEP COUNT only. Probe points live nowhere else, so a
    wholly lost log still fails -- see test_probeless_log_is_not_rescued below.
    """
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _probe_output(tmp, m, [
            {"global_step": 0, "phase": "begin", "rank_acc": 0.5, "margin_mean": 0.0},
            {"global_step": 644, "phase": "end", "rank_acc": 0.9, "margin_mean": 0.3},
        ])
        # every step record lost to EREMOTEIO; only the probes landed
        kept = [l for l in (out / TRAINING_LOG_NAME).read_text().splitlines()
                if '"phase"' in l]
        (out / TRAINING_LOG_NAME).write_text("\n".join(kept) + "\n")
        ckpt = _ckpt(out, step=644)
        _safetensors(ckpt / "model.safetensors", 1)
        (ckpt / "config.json").write_text(json.dumps({"model_type": "xlm-roberta"}))
        result = assert_training_succeeded(out, m)
        assert result['final_global_step'] == 644, result


def test_probeless_log_is_not_rescued():
    """trainer_state.json carries steps, not ranking signal. Two points are required."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _trained_output(tmp, m)
        (out / TRAINING_LOG_NAME).unlink()
        ckpt = _ckpt(out, step=644)
        _safetensors(ckpt / "model.safetensors", 1)
        (ckpt / "config.json").write_text(json.dumps({"model_type": "xlm-roberta"}))
        _assert_raises(RunDirectoryError,
                       lambda: assert_training_succeeded(out, m), "ranking probe")


def test_unreadable_manifest_raises_rather_than_advising_overwrite():
    """An IO error is not a config mismatch; --overwrite would delete checkpoints."""
    import builtins
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = Path(tmp) / "out"
        prepare_output_dir(out, m)
        _ckpt(out, step=500)

        real_open, real_sleep = builtins.open, time.sleep
        real_read = Path.read_text

        def boom(self, *a, **kw):
            if self.name == RUN_MANIFEST_NAME:
                raise OSError(121, "Remote I/O error")
            return real_read(self, *a, **kw)

        Path.read_text, time.sleep = boom, lambda *_: None
        try:
            msg = _assert_raises(RunDirectoryError,
                                 lambda: prepare_output_dir(out, _manifest(tmp)))
        finally:
            Path.read_text, builtins.open, time.sleep = real_read, real_open, real_sleep
        assert "do NOT pass --overwrite" in msg, msg
        assert list(out.glob("checkpoint-*")), "checkpoints must survive"


def test_malformed_manifest_json_still_blocks():
    """Corrupt JSON is genuinely unidentifiable state: that branch is unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        prepare_output_dir(out, _manifest(tmp))
        (out / RUN_MANIFEST_NAME).write_text("{not json")
        _assert_raises(RunDirectoryError,
                       lambda: prepare_output_dir(out, _manifest(tmp)),
                       "--overwrite")


# ---- invocation progress: resume must take NEW steps -------------------------

def test_resume_with_zero_new_steps_fails():
    """The resume path used to re-open the zero-step hole the gate exists to close.

    --resume keeps training_log.jsonl, so max(global_step) is the PREVIOUS run's
    final step. Without an invocation start step, a resumed run that trains nothing
    reads 2062 and reports success.
    """
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _trained_output(tmp, m, steps=2062)
        ckpt = _ckpt(out, step=2062)
        (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 2062}))
        _safetensors(ckpt / "model.safetensors", 1)
        (ckpt / "config.json").write_text(json.dumps({"model_type": "xlm-roberta"}))

        resumed = _manifest(tmp)
        prepare_output_dir(out, resumed, resume=True)
        assert resumed.get('invocation_start_step') == 2062, resumed.get('invocation_start_step')
        _assert_raises(RunDirectoryError,
                       lambda: assert_training_succeeded(out, resumed),
                       "no new optimizer steps")


def test_resume_refuses_when_start_step_indeterminate():
    """A checkpoint with no readable trainer_state.json gives no progress baseline."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = Path(tmp) / "out"
        prepare_output_dir(out, m)
        _ckpt(out, step=2062, state=False)     # HF state file lost/never written
        _assert_raises(RunDirectoryError,
                       lambda: prepare_output_dir(out, _manifest(tmp), resume=True),
                       "invocation start step")


def test_run_short_of_planned_steps_fails():
    """A wall-clock kill at 60% must not report success."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp, optimizer_steps=1000)
        out = _trained_output(tmp, m, steps=600)
        _assert_raises(RunDirectoryError,
                       lambda: assert_training_succeeded(out, m),
                       "1000")


def test_undeletable_checkpoint_raises():
    """rmtree(ignore_errors=True) hid EREMOTEIO; a survivor means Tevatron resumes."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = Path(tmp) / "out"
        prepare_output_dir(out, m)
        _ckpt(out, step=17202)

        real_rmtree, real_sleep = shutil.rmtree, time.sleep
        shutil.rmtree, time.sleep = (lambda *a, **k: None), (lambda *_: None)
        try:
            msg = _assert_raises(RunDirectoryError,
                                 lambda: prepare_output_dir(out, _manifest(tmp)))
        finally:
            shutil.rmtree, time.sleep = real_rmtree, real_sleep
        assert "checkpoint-17202" in msg, msg


# ---- probes must be successful, at two distinct steps -----------------------

def _probe_output(tmp, manifest, probes):
    out = Path(tmp) / "out"
    prepare_output_dir(out, manifest)
    append_jsonl(out / TRAINING_LOG_NAME,
                 {"global_step": 100, "loss": 0.5, "learning_rate": 1e-5, "grad_norm": 1.2})
    for rec in probes:
        append_jsonl(out / TRAINING_LOG_NAME, rec)
    (out / "config.json").write_text(json.dumps({"model_type": "xlm-roberta"}))
    _safetensors(out / "model.safetensors", 1)
    return out


def test_error_only_probes_fail():
    """In-batch 14990 shipped {"phase":"begin","error":"TypeError…"} and still passed."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _probe_output(tmp, m, [
            {"global_step": 0, "phase": "begin", "error": "TypeError: x"},
            {"global_step": 100, "phase": "end", "error": "TypeError: x"},
        ])
        _assert_raises(RunDirectoryError,
                       lambda: assert_training_succeeded(out, m), "ranking probe")


def test_duplicate_step_probes_fail():
    """Two successful probes at the SAME step are one point, not two."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _probe_output(tmp, m, [
            {"global_step": 0, "phase": "begin", "rank_acc": 0.5, "margin_mean": 0.0},
            {"global_step": 0, "phase": "end", "rank_acc": 0.9, "margin_mean": 0.3},
        ])
        _assert_raises(RunDirectoryError,
                       lambda: assert_training_succeeded(out, m), "ranking probe")


def test_non_finite_probe_does_not_count():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _probe_output(tmp, m, [
            {"global_step": 0, "phase": "begin", "rank_acc": 0.5, "margin_mean": 0.0},
            {"global_step": 100, "phase": "end", "rank_acc": float('nan'), "margin_mean": 0.3},
        ])
        _assert_raises(RunDirectoryError,
                       lambda: assert_training_succeeded(out, m), "ranking probe")


def test_two_distinct_successful_probes_pass():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = _probe_output(tmp, m, [
            {"global_step": 0, "phase": "begin", "rank_acc": 0.5, "margin_mean": 0.0},
            {"global_step": 100, "phase": "end", "rank_acc": 0.9, "margin_mean": 0.3},
        ])
        result = assert_training_succeeded(out, m)
        assert result['final_global_step'] == 100, result



# ---- final-batch shape: the docs make a claim about this --------------------

def test_final_batch_sizes_match_the_documented_pools():
    """docs/inbatch.md and docs/crossbatch.md state different final-batch behaviour.

    drop_last is unset (HF default False), so in-batch really does end each epoch on
    a 9-query batch (17 negatives). Cross-batch does NOT: Accelerate's default
    even_batches pads the distributed sampler so every rank gets an equal batch, so
    the pool stays 2,047 and only the count of NEW records falls.
    """
    n = 329993
    full, rem = divmod(n, 64)
    assert (full, rem) == (5156, 9), (full, rem)
    assert rem * 2 - 1 == 17

    full_cb, rem_cb = divmod(n, 1024)
    assert (full_cb, rem_cb) == (322, 265), (full_cb, rem_cb)

    # Accelerate pads to a whole number of per-rank batches across 2 ranks. The
    # padded step is a full 1024 queries -> 2048 passages -> 2047 negatives.
    world, per_device = 2, 512
    padded = -(-rem_cb // (per_device * world)) * (per_device * world)
    assert padded == 1024, padded
    assert padded * 2 - 1 == 2047

    # The arithmetic above is the invariant. The prose below is thesis-writing
    # material kept out of version control, so check it only when it is present
    # locally -- a clone without it must still pass.
    docs = (project_root / 'docs')
    ib_p, cb_p = docs / 'inbatch.md', docs / 'crossbatch.md'
    if ib_p.is_file() and cb_p.is_file():
        ib, cb = ib_p.read_text(), cb_p.read_text()
        assert "17 negatives" in ib and "9 queries" in ib, "in-batch final batch undocumented"
        assert "529" not in cb, "cross-batch must not claim a shrunken final pool"
        assert "265 new" in cb and "2,047" in cb


def test_even_batches_is_the_pinned_accelerate_default():
    """The padding claim rests on this default; a change would invalidate the docs."""
    from accelerate.utils import DataLoaderConfiguration
    assert DataLoaderConfiguration().even_batches is True



def test_undeletable_stale_log_raises():
    """Removal is CLAIMED, so it must be verified -- same as the checkpoints above.

    Reproduced: with unlink failing permanently, a fresh run inherited the previous
    run's 100 steps and 2 probes and validated after taking zero steps of its own.
    """
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp)
        out = Path(tmp) / "out"
        prepare_output_dir(out, m)
        append_jsonl(out / TRAINING_LOG_NAME, {"global_step": 3000, "loss": 0.4})

        real_unlink, real_sleep = Path.unlink, time.sleep
        Path.unlink = lambda self, *a, **k: None      # silently does nothing
        time.sleep = lambda *_: None
        try:
            msg = _assert_raises(RunDirectoryError,
                                 lambda: prepare_output_dir(out, _manifest(tmp)))
        finally:
            Path.unlink, time.sleep = real_unlink, real_sleep
        assert TRAINING_LOG_NAME in msg, msg


def test_corrupt_manifest_is_not_mistaken_for_a_legacy_checkpoint():
    """A malformed manifest must not silently disable the encoding-contract check."""
    from utils.helpers import load_training_manifest
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "model"
        d.mkdir()
        assert load_training_manifest(d) is None, "absent manifest is legitimately None"
        (d / RUN_MANIFEST_NAME).write_text("{not json")
        _assert_raises(RunDirectoryError, lambda: load_training_manifest(d),
                       "could not be read")


def test_unreadable_eval_manifest_is_not_mistaken_for_legacy():
    """A present manifest lost to transient IO must fail closed after retries."""
    from utils.helpers import load_training_manifest
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "model"
        d.mkdir()
        manifest = d / RUN_MANIFEST_NAME
        manifest.write_text("{}")

        real_read_text, real_sleep = Path.read_text, time.sleep

        def fail_manifest_read(self, *args, **kwargs):
            if self == manifest:
                raise OSError("remote IO")
            return real_read_text(self, *args, **kwargs)

        Path.read_text, time.sleep = fail_manifest_read, lambda *_: None
        try:
            _assert_raises(RunDirectoryError, lambda: load_training_manifest(d),
                           "could not read")
        finally:
            Path.read_text, time.sleep = real_read_text, real_sleep


TESTS = [
    ("dir: stale checkpoints removed on a fresh run", test_stale_checkpoints_removed_on_fresh_run),
    ("dir: unfinished run does not block a corrected re-run", test_unfinished_run_does_not_block_a_corrected_rerun),
    ("dir: finished run still blocks a different config", test_finished_run_still_blocks_a_different_config),
    ("dir: unidentifiable checkpoints refused, not deleted", test_unidentifiable_checkpoints_are_refused_not_deleted),
    ("dir: clean dir with no manifest proceeds", test_clean_dir_with_no_manifest_still_proceeds),
    ("dir: same fingerprint re-runs cleanly", test_fresh_run_over_same_fingerprint_is_allowed),
    ("dir: different fingerprint refuses", test_different_fingerprint_refuses),
    ("dir: overwrite permits an incompatible dir", test_overwrite_permits_an_incompatible_dir),
    ("dir: resume without a prior manifest refuses", test_resume_without_a_prior_manifest_refuses),
    ("dir: resume with an incompatible manifest refuses", test_resume_with_an_incompatible_manifest_refuses),
    ("dir: resume keeps checkpoints", test_resume_keeps_checkpoints),
    ("dir: resume without a checkpoint refuses", test_resume_without_a_checkpoint_refuses),
    ("dir: unreadable manifest is incompatible, not absent", test_unreadable_manifest_is_incompatible_not_absent),
    ("manifest: fingerprint is stable", test_fingerprint_is_stable_across_two_builds),
    ("manifest: fingerprint moves with config/data/world size", test_fingerprint_moves_with_config_data_and_world_size),
    ("manifest: provenance recorded without raising", test_manifest_records_provenance_without_raising),
    ("success: zero optimizer steps fails", test_zero_optimizer_steps_fails),
    ("success: non-finite loss fails", test_non_finite_loss_fails),
    ("success: missing diagnostics fails", test_missing_diagnostics_fails),
    ("success: stale checkpoint fails", test_stale_checkpoint_fails),
    ("success: truncated checkpoint fails", test_truncated_checkpoint_fails),
    ("success: appends final step and probes to the manifest", test_success_appends_to_the_manifest),
    ("probe: margin and rank_acc are exact", test_probe_margin_and_rank_acc_are_exact),
    ("probe: None tokenizer refused clearly", test_probe_refuses_a_none_tokenizer_with_a_clear_message),
    ("probe: entry points supply their own tokenizer", test_entry_points_supply_their_own_probe_tokenizer),
    ("probe: deterministic", test_probe_is_deterministic),
    ("probe: restores train mode", test_probe_restores_train_mode),
    ("probe: fixed to the last train_hq records", test_probe_triples_are_the_last_records_of_train_hq),
    ("probe: skips records without a negative", test_probe_triples_skip_records_without_a_negative),
    ("callback: persists loss/LR/pre-clipping grad norm", test_callback_persists_loss_lr_and_pre_clipping_grad_norm),
    ("callback: probes at three points", test_callback_probes_at_three_points),
    ("callback: a failing probe is recorded, not fatal", test_callback_records_a_failing_probe_instead_of_killing_the_run),
    ("callback: registration is idempotent", test_callback_registration_is_idempotent),
    ("crossbatch: non-reentrant checkpointing for DDP+GradCache", test_crossbatch_uses_non_reentrant_checkpointing),
    ("pool: world size 2 gives 1024/2048/2047", test_pool_world_size_two_gives_1024_2048_2047),
    ("pool: single process refuses", test_pool_single_process_refuses),
    ("pool: gradient accumulation refuses", test_pool_accumulation_refuses),
    ("pool: grad_cache off refuses", test_pool_grad_cache_off_refuses),
    ("config: unused key refuses", test_config_unused_key_refuses),
    ("config: missing key refuses", test_config_missing_key_refuses),
    ("config: optional keys tolerated", test_config_optional_keys_may_be_absent_or_present),
    ("config: live recipes satisfy the entry points", test_real_recipes_agree_with_the_entry_points),
    ("log: append survives a failing open", test_append_jsonl_survives_a_failing_open),
    ("log: append retries then succeeds", test_append_jsonl_retries_then_succeeds),
    ("dir: fresh run clears a stale training log", test_fresh_run_clears_a_stale_training_log),
    ("dir: resume keeps the training log", test_resume_keeps_the_training_log),
    ("success: grad_norm warns, loss raises", test_non_finite_grad_norm_warns_but_loss_still_raises),
    ("success: truncated last line tolerated", test_truncated_last_log_line_tolerated),
    ("success: lost step records fall back to trainer_state", test_lost_step_records_fall_back_to_trainer_state),
    ("success: probeless log is not rescued", test_probeless_log_is_not_rescued),
    ("dir: unreadable manifest does not advise --overwrite",
     test_unreadable_manifest_raises_rather_than_advising_overwrite),
    ("dir: malformed manifest JSON still blocks", test_malformed_manifest_json_still_blocks),
    ("guard: resume with zero new steps fails", test_resume_with_zero_new_steps_fails),
    ("guard: resume refuses an indeterminate start", test_resume_refuses_when_start_step_indeterminate),
    ("guard: run short of planned steps fails", test_run_short_of_planned_steps_fails),
    ("guard: undeletable checkpoint raises", test_undeletable_checkpoint_raises),
    ("guard: undeletable stale log raises", test_undeletable_stale_log_raises),
    ("provenance: corrupt manifest is not legacy", test_corrupt_manifest_is_not_mistaken_for_a_legacy_checkpoint),
    ("provenance: unreadable manifest is not legacy", test_unreadable_eval_manifest_is_not_mistaken_for_legacy),
    ("probe: error-only probes fail", test_error_only_probes_fail),
    ("probe: duplicate-step probes fail", test_duplicate_step_probes_fail),
    ("probe: non-finite probe does not count", test_non_finite_probe_does_not_count),
    ("probe: two distinct successful probes pass", test_two_distinct_successful_probes_pass),
    ("batch: final-batch sizes match the docs", test_final_batch_sizes_match_the_documented_pools),
    ("batch: even_batches default is True", test_even_batches_is_the_pinned_accelerate_default),
]


def _run(name, fn):
    try:
        fn()
    except Exception as e:                                        # noqa: BLE001
        print(f"  ❌ {name}\n       {type(e).__name__}: {e}")
        if os.environ.get("TEST_TRACE"):
            traceback.print_exc()
        return False
    print(f"  ✅ {name}")
    return True


def main():
    print("\nTraining guard tests (in-batch / cross-batch)")
    print("=" * 58)
    passed = sum(_run(n, f) for n, f in TESTS)
    print("=" * 58)
    print(f"  {passed}/{len(TESTS)} passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
