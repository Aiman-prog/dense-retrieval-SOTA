"""ANCE handoff, supervision, freshness and success validation.

The defects these pin, all of which shipped:

* the trainer started at `last_ann_no = 0` while the inferencer numbered from
  `get_latest_marker_no(ann_dir) + 1` in a work root shared by every run, so a
  leftover `ready_7` was swapped in at the first logging step;
* the inferencer was an unsupervised `Popen`, so a run whose miner died trained to
  `max_steps` on base-model negatives and exited 0 as "ANCE";
* nothing checked the loss for NaN and nothing called `assert_training_succeeded`.

Run: python tests/ance_pipeline_test.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.helpers import (                                       # noqa: E402
    RUN_MANIFEST_NAME, TRAINING_LOG_NAME, RunDirectoryError, append_jsonl,
    assert_training_succeeded, build_run_manifest, prepare_output_dir,
)
from ance_mining import (                                         # noqa: E402
    INITIAL_ROUND, RoundError, assert_ance_refresh, latest_committed_round,
    publish_round, read_round, round_paths,
)


def _assert_raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        assert contains is None or contains in str(e), str(e)
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


RUN_ID = "abc123def456-1700000000"


def _records(n=3, prefix="q"):
    return [{'query_id': f'{prefix}{i}', 'query': f'query {i}',
             'positive_passages': [{'docid': f'p{i}', 'text': f'pos {i}'}],
             'negative_passages': [{'docid': f'n{i}', 'text': f'neg {i}'}]}
            for i in range(n)]


def _publish(root, n, *, run_id=RUN_ID, step=1000, failures=0, records=None):
    return publish_round(
        root, n,
        records_by_file=[("train_hq.jsonl", records if records is not None
                          else _records())],
        meta={'run_id': run_id, 'ann_no': n,
              'checkpoint': f'/models/x/checkpoint-{step}', 'checkpoint_step': step,
              'n_queries_mined': 3, 'n_sampling_failures': failures})


# ---- round commit protocol --------------------------------------------------

def test_marker_is_written_last_and_metadata_first():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _publish(root, 1)
        assert paths['meta'].is_file() and paths['ready'].is_file()
        meta = json.loads(paths['meta'].read_text())
        assert sorted(meta['file_sha256']) == meta['files']
        # marker mtime is never older than the metadata it vouches for
        assert paths['ready'].stat().st_mtime >= paths['meta'].stat().st_mtime
        assert not paths['work'].exists(), "staging directory left behind"


def test_a_round_without_its_marker_is_invisible():
    """A crash between the rename sequence and the marker leaves final-path
    artifacts. Those are not a committed round."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _publish(root, 1)
        _publish(root, 2)
        round_paths(root, 2)['ready'].unlink()
        assert latest_committed_round(root) == 1
        _assert_raises(RoundError, lambda: read_round(root, 2, run_id=RUN_ID),
                       "no ready_2 marker")


def test_marker_without_metadata_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _publish(root, 1)
        round_paths(root, 1)['meta'].unlink()
        _assert_raises(RoundError, lambda: read_round(root, 1, run_id=RUN_ID),
                       "provenance cannot be established")


def test_a_foreign_runs_round_is_refused():
    """The core E1 case: a leftover round from a previous run used to be consumed
    at the first logging step because the trainer only compared round numbers."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _publish(root, 7, run_id="OTHER-RUN-9999")
        assert latest_committed_round(root) == 7, "still discoverable by number"
        msg = _assert_raises(RoundError, lambda: read_round(root, 7, run_id=RUN_ID),
                             "not this experiment")
        assert "OTHER-RUN-9999" in msg and RUN_ID in msg, msg


def test_truncated_round_data_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _publish(root, 1)
        target = paths['training_data'] / "train_hq.jsonl"
        lines = target.read_text().splitlines()
        target.write_text("\n".join(lines[:-1]) + "\n")
        _assert_raises(RoundError, lambda: read_round(root, 1, run_id=RUN_ID),
                       "content hash")


def test_same_count_round_mutation_is_refused():
    """A rewrite can preserve the line count, so counts alone are not integrity."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _publish(root, 1)
        target = paths['training_data'] / "train_hq.jsonl"
        original = target.read_text()
        target.write_text(original.replace('query 0', 'query X', 1))
        assert len(target.read_text().splitlines()) == len(original.splitlines())
        _assert_raises(RoundError, lambda: read_round(root, 1, run_id=RUN_ID),
                       "content hash")


def test_extra_file_in_a_round_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _publish(root, 1)
        (paths['training_data'] / "stray.jsonl").write_text('{"query_id":"z"}\n')
        _assert_raises(RoundError, lambda: read_round(root, 1, run_id=RUN_ID),
                       "do not match the metadata")


def test_missing_data_directory_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = _publish(root, 1)
        shutil.rmtree(paths['training_data'])
        _assert_raises(RoundError, lambda: read_round(root, 1, run_id=RUN_ID),
                       "is missing")


def test_sampling_failures_block_publication_entirely():
    """No marker, no data, no metadata: a round that could not supply its ANN
    negatives is discarded rather than published with substitutes."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _assert_raises(RoundError, lambda: _publish(root, 1, failures=1),
                       "never fabricates a negative")
        assert latest_committed_round(root) == 0
        assert not round_paths(root, 1)['ready'].exists()
        assert not round_paths(root, 1)['training_data'].exists()


def test_initial_round_is_not_counted_as_a_numeric_round():
    """`ready_initial` is the step-0 input, not a refresh."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _publish(root, INITIAL_ROUND, step=0)
        assert latest_committed_round(root) == 0
        data_dir, meta = read_round(root, INITIAL_ROUND, run_id=RUN_ID)
        assert meta['checkpoint_step'] == 0
        assert data_dir.name == "training_data_initial"


def test_latest_committed_round_tracks_the_highest_marker():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _publish(root, INITIAL_ROUND, step=0)
        assert latest_committed_round(root) == 0
        for n in (1, 2, 3):
            _publish(root, n, step=1000 * n)
            assert latest_committed_round(root) == n


# ---- freshness gate ---------------------------------------------------------

def _summary(*rounds):
    return {'run_id': RUN_ID, 'rounds': list(rounds)}


def _round(ann_no, step, consumed):
    return {'ann_no': ann_no, 'checkpoint': f'ck-{step}',
            'checkpoint_step': step, 'consumed_steps': consumed}


def test_initial_round_alone_does_not_satisfy_the_gate():
    """The exact shape of a run whose inferencer died at startup."""
    _assert_raises(
        RoundError,
        lambda: assert_ance_refresh(_summary(_round(INITIAL_ROUND, 0, 10312))),
        "is not ANCE")


def test_no_rounds_at_all_does_not_satisfy_the_gate():
    _assert_raises(RoundError, lambda: assert_ance_refresh(_summary()), "none")


def test_a_published_but_never_consumed_round_does_not_count():
    """Mined at step 10000 of 10312 and swapped in with no steps left to train on
    it: the negatives never influenced a single update."""
    _assert_raises(
        RoundError,
        lambda: assert_ance_refresh(_summary(_round(INITIAL_ROUND, 0, 10000),
                                             _round(1, 1000, 0))),
        "0 qualified")


def test_one_consumed_checkpoint_round_satisfies_the_gate():
    fresh = assert_ance_refresh(_summary(_round(INITIAL_ROUND, 0, 900),
                                         _round(1, 1000, 412)))
    assert [r['ann_no'] for r in fresh] == [1]


def test_min_consume_steps_is_enforced():
    summary = _summary(_round(INITIAL_ROUND, 0, 900), _round(1, 1000, 5))
    assert_ance_refresh(summary, min_consume_steps=5)
    _assert_raises(RoundError,
                   lambda: assert_ance_refresh(summary, min_consume_steps=6))


def test_min_fresh_rounds_is_enforced():
    summary = _summary(_round(1, 1000, 100), _round(2, 2000, 100))
    assert len(assert_ance_refresh(summary, min_fresh_rounds=2)) == 2
    _assert_raises(RoundError,
                   lambda: assert_ance_refresh(summary, min_fresh_rounds=3))


# ---- inferencer supervision -------------------------------------------------

class _Proc:
    """Minimal Popen stand-in.

    `rc` is what poll() reports once the process is no longer alive. `exit_after`
    is how many timed-out waits it survives before exiting, which is what lets the
    supervision loop terminate in a test.
    """

    def __init__(self, rc=None, alive=True, exit_after=1):
        self._rc, self.alive, self._exit_after = rc, alive, exit_after
        self.returncode = None if alive else rc
        self.terminated = self.killed = False
        self.waits = 0

    def poll(self):
        return None if self.alive else self._rc

    def wait(self, timeout=None):
        self.waits += 1
        if self.alive and timeout is not None and self.waits <= self._exit_after:
            import subprocess
            raise subprocess.TimeoutExpired("proc", timeout)
        self.alive = False
        self.returncode = self._rc
        return self._rc

    def terminate(self):
        self.terminated = True
        self.alive = False
        self.returncode = self._rc

    def kill(self):
        self.killed = True


def _supervise(*a, **kw):
    from train_ance import supervise
    kw.setdefault('log', lambda *_: None)
    return supervise(*a, poll_seconds=0.0, **kw)


def test_a_dead_inferencer_kills_the_trainer_and_fails_the_run():
    trainer = _Proc(rc=0, alive=True)
    inferencer = _Proc(rc=1, alive=False)
    failed, train_rc = _supervise(trainer, inferencer)
    assert failed == 1, failed
    assert trainer.terminated, "the trainer was left running on stale negatives"


def test_an_inferencer_that_dies_with_the_trainer_is_still_caught():
    """The window the in-loop check alone would miss."""
    trainer = _Proc(rc=0, alive=True)
    inferencer = _Proc(rc=137, alive=True, exit_after=99)

    original = trainer.wait

    def wait(timeout=None):
        inferencer.alive = False           # dies in the same window
        inferencer.returncode = 137
        return original(timeout=None)

    trainer.wait = wait
    failed, _ = _supervise(trainer, inferencer)
    assert failed == 137, failed


def test_a_healthy_run_reports_no_failure():
    trainer = _Proc(rc=0, alive=True, exit_after=2)
    inferencer = _Proc(rc=None, alive=True, exit_after=99)
    failed, train_rc = _supervise(trainer, inferencer)
    assert failed is None, failed
    assert train_rc == 0
    assert inferencer.terminated, "the inferencer must be stopped when training ends"


def test_a_clean_early_inferencer_exit_is_a_failure():
    """INVERTED. `run_ance_data_gen.main()` loops until terminated and has no
    --max_rounds equivalent, so there is no path on which it finishes early and
    legitimately. rc 0 before the trainer is done means refreshes stopped."""
    trainer = _Proc(rc=0, alive=True)
    inferencer = _Proc(rc=0, alive=False)
    failed, _ = _supervise(trainer, inferencer)
    assert failed == 0, failed          # 0 is a failure code here, not "no failure"
    assert failed is not None, "rc 0 must not be read as truthiness"
    assert trainer.terminated, "the trainer was left running with no miner"


def test_our_own_termination_is_not_an_early_exit():
    """After the trainer finishes we terminate the inferencer ourselves. That is a
    normal shutdown and must not be reported as a failed run."""
    trainer = _Proc(rc=0, alive=True, exit_after=1)
    inferencer = _Proc(rc=-15, alive=True, exit_after=99)
    failed, train_rc = _supervise(trainer, inferencer)
    assert failed is None, failed
    assert train_rc == 0
    assert inferencer.terminated


# ---- non-finite loss --------------------------------------------------------

def test_non_finite_loss_is_rejected_before_backward():
    from run_ance_train import NonFiniteLoss, check_finite_loss
    assert check_finite_loss(0.42, 10) == 0.42
    for bad in (float('nan'), float('inf'), float('-inf')):
        _assert_raises(NonFiniteLoss, lambda b=bad: check_finite_loss(b, 10),
                       "diverged")


# ---- success validation -----------------------------------------------------

_RECIPE = {'model_name': 'm', 'batch_size': 4, 'train_group_size': 2,
           'learning_rate': 1e-5, 'total_epochs': 2}
_CTX = {'args': _RECIPE, 'base_model': '/nonexistent/base'}


def _mixture(tmp):
    path = Path(tmp) / "train_hq.jsonl"
    with open(path, 'w') as f:
        for rec in _records(5):
            f.write(json.dumps(rec) + '\n')
    return path


def _manifest(tmp, steps=10):
    return build_run_manifest('ance', _CTX, _RECIPE, data_files=[_mixture(tmp)],
                              world_size=1, negative_pool_size=7,
                              optimizer_steps=steps)


def _safetensors(path):
    header = {"t0": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, 'little') + raw + b"\x00" * 4)


def _trained_output(tmp, manifest, *, logged, final, planned=10):
    """An output dir shaped like the ANCE trainer's own writes.

    `logged` are the steps at which the loop's logging_steps branch fired; `final`
    is the terminal record. There is no trainer_state.json: this is not an HF
    Trainer, so the diagnostics log is the ONLY witness.
    """
    out = Path(tmp) / "out"
    prepare_output_dir(out, manifest)
    log = out / TRAINING_LOG_NAME
    append_jsonl(log, {"global_step": 0, "phase": "begin",
                       "rank_acc": 0.5, "margin_mean": 0.0})
    for step in logged:
        append_jsonl(log, {"global_step": step, "loss": 0.5,
                           "learning_rate": 1e-5, "grad_norm": 1.2})
    if final is not None:
        append_jsonl(log, {"global_step": final, "loss": 0.4, "learning_rate": 1e-5,
                           "grad_norm": 1.1, "terminal": True})
    append_jsonl(log, {"global_step": final or (logged[-1] if logged else 0),
                       "phase": "end", "rank_acc": 0.9, "margin_mean": 0.3})
    (out / "config.json").write_text(json.dumps({"model_type": "xlm-roberta"}))
    _safetensors(out / "model.safetensors")
    return out


def test_a_complete_run_validates_via_the_terminal_record():
    """max_steps is not a multiple of logging_steps. Without the terminal record the
    log's last step is 8 against a planned 10 and a COMPLETE run is rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp, steps=10)
        out = _trained_output(tmp, m, logged=[4, 8], final=10)
        stored = assert_training_succeeded(out, m)
        assert stored['final_global_step'] == 10


def test_without_the_terminal_record_a_complete_run_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp, steps=10)
        out = _trained_output(tmp, m, logged=[4, 8], final=None)
        _assert_raises(RunDirectoryError,
                       lambda: assert_training_succeeded(out, m),
                       "stopped at step 8 of the 10 planned")


def test_a_zero_step_run_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp, steps=10)
        out = _trained_output(tmp, m, logged=[], final=None)
        _assert_raises(RunDirectoryError,
                       lambda: assert_training_succeeded(out, m),
                       "no new optimizer steps")


def test_a_non_finite_loss_in_the_log_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp, steps=10)
        out = _trained_output(tmp, m, logged=[4, 8], final=10)
        append_jsonl(out / TRAINING_LOG_NAME,
                     {"global_step": 6, "loss": float('nan'), "grad_norm": 1.0})
        _assert_raises(RunDirectoryError,
                       lambda: assert_training_succeeded(out, m), "non-finite loss")


def test_manifest_records_the_run_identity_for_the_evaluator():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manifest(tmp, steps=10)
        out = _trained_output(tmp, m, logged=[4, 8], final=10)
        assert_training_succeeded(out, m)
        stored = json.loads((out / RUN_MANIFEST_NAME).read_text())
        assert stored['finished_at'] and stored['fingerprint']
        assert stored['optimizer_steps_planned'] == 10


# ---- non-finite gradients ---------------------------------------------------

def _tiny_model():
    """A real 2-layer encoder wrapped in the real DenseModel, on CPU."""
    from transformers import XLMRobertaConfig, XLMRobertaModel
    from tevatron.retriever.modeling import DenseModel
    cfg = XLMRobertaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=64,
                           max_position_embeddings=12)
    import torch as t
    t.manual_seed(0)
    return DenseModel(encoder=XLMRobertaModel(cfg), pooling='cls', normalize=True,
                      temperature=0.02)


def _grad_fixture(value):
    """A model whose every gradient is `value`, plus a real AdamW and scheduler."""
    import torch as t
    from transformers import get_linear_schedule_with_warmup
    from utils.helpers import build_adamw
    model = _tiny_model()
    optimizer, _ = build_adamw(model.parameters(), lr=1e-3, weight_decay=0.0,
                               label='test')
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, 10)
    for prm in model.parameters():
        prm.grad = t.full_like(prm, value)
    before = [prm.detach().clone() for prm in model.parameters()]
    return model, optimizer, scheduler, before


def test_non_finite_grad_norm_is_rejected():
    from run_ance_train import NonFiniteGradNorm, check_finite_grad_norm
    assert check_finite_grad_norm(1.5, 10) == 1.5
    for bad in (float('nan'), float('inf'), float('-inf')):
        _assert_raises(NonFiniteGradNorm,
                       lambda b=bad: check_finite_grad_norm(b, 10),
                       "clipping cannot rescue")


def test_apply_gradients_does_not_step_on_non_finite_gradients():
    """The ordering claim, proved by state: parameters must be untouched after the
    raise. clip_grad_norm_ with a non-finite total norm yields a non-finite
    coefficient, so a step here writes NaN into every parameter."""
    import torch as t
    from run_ance_train import NonFiniteGradNorm, apply_gradients
    model, optimizer, scheduler, before = _grad_fixture(float('inf'))
    _assert_raises(NonFiniteGradNorm,
                   lambda: apply_gradients(model, optimizer, scheduler,
                                           max_grad_norm=1.0, step=7),
                   "step 7")
    for prm, was in zip(model.parameters(), before):
        assert t.equal(prm.detach(), was), "a parameter moved despite the raise"
        assert t.isfinite(prm.detach()).all(), "NaN reached the weights"


def test_apply_gradients_steps_on_finite_gradients():
    import torch as t
    from run_ance_train import apply_gradients
    model, optimizer, scheduler, before = _grad_fixture(0.01)
    norm = apply_gradients(model, optimizer, scheduler, max_grad_norm=1.0, step=1)
    assert norm > 0 and t.isfinite(t.tensor(norm))
    moved = sum(not t.equal(prm.detach(), was)
                for prm, was in zip(model.parameters(), before))
    assert moved > 0, "no parameter moved on a valid step"
    assert all(prm.grad is None for prm in model.parameters()), "grads not zeroed"


def test_a_finite_loss_does_not_imply_finite_gradients():
    """Why the loss guard alone is not enough: they are independent checks."""
    from run_ance_train import (NonFiniteGradNorm, apply_gradients,
                                check_finite_loss)
    assert check_finite_loss(0.5, 3) == 0.5          # forward pass looks healthy
    model, optimizer, scheduler, _ = _grad_fixture(float('nan'))
    _assert_raises(NonFiniteGradNorm,
                   lambda: apply_gradients(model, optimizer, scheduler,
                                           max_grad_norm=1.0, step=3))


# ---- the trainer summary is critical, not best-effort -----------------------

def test_summary_write_is_retried_then_raises():
    """`ance_trainer_summary.json` is the only evidence of round consumption, and
    train_ance.py fails the run when it is missing. A lost write must raise, not
    leave the run unvalidatable."""
    import utils.helpers as helpers
    calls = {'n': 0}

    def always_fails():
        calls['n'] += 1
        raise OSError("EREMOTEIO")

    assert helpers.retry_io(always_fails, "write summary", attempts=3, delay=0) is False
    assert calls['n'] == 3, calls['n']


def test_summary_write_survives_a_transient_failure():
    import utils.helpers as helpers
    calls = {'n': 0}

    def fails_twice():
        calls['n'] += 1
        if calls['n'] < 3:
            raise OSError("EREMOTEIO")

    assert helpers.retry_io(fails_twice, "write summary", attempts=5, delay=0) is True
    assert calls['n'] == 3


def test_trainer_treats_a_failed_summary_write_as_fatal():
    """Source-level: the write must be guarded by retry_io AND raise on False."""
    src = (project_root / 'scripts' / 'run_ance_train.py').read_text()
    body = src[src.index("def _write_summary("):src.index("if args.max_steps < 1:")]
    assert 'retry_io(' in body, "the summary write is still best-effort"
    assert 'raise OSError' in body, "a failed summary write does not raise"


# ---- run id -----------------------------------------------------------------

def _fake_manifest(fp="abcdef0123456789", epoch=1700000000.0):
    return {'fingerprint': fp, 'started_at_epoch': epoch}


def test_run_ids_are_unique_within_one_second():
    """A SLURM array launches its tasks in the same second with the same recipe, so
    the fingerprint and the timestamp are both identical across them."""
    from train_ance import build_run_id
    m = _fake_manifest()
    ids = {build_run_id(m) for _ in range(200)}
    assert len(ids) == 200, f"{200 - len(ids)} collision(s)"


def test_run_id_carries_the_fingerprint_and_job_id():
    from train_ance import build_run_id
    os.environ['SLURM_JOB_ID'] = '9566838'
    try:
        rid = build_run_id(_fake_manifest())
    finally:
        del os.environ['SLURM_JOB_ID']
    assert rid.startswith('abcdef012345'), rid
    assert 'j9566838' in rid, rid


def test_run_id_omits_the_job_id_outside_slurm():
    from train_ance import build_run_id
    os.environ.pop('SLURM_JOB_ID', None)
    assert 'j' not in build_run_id(_fake_manifest()).split('-')[-1]


def test_work_root_creation_refuses_an_existing_directory():
    """exist_ok=False is what makes a collision a startup error instead of two runs
    silently sharing a work root, each refusing the other's rounds."""
    src = (project_root / 'scripts' / 'train_ance.py').read_text()
    assert 'work_root.mkdir(parents=True, exist_ok=False)' in src
    with tempfile.TemporaryDirectory() as tmp:
        existing = Path(tmp) / "collide"
        existing.mkdir()
        _assert_raises(FileExistsError,
                       lambda: existing.mkdir(parents=True, exist_ok=False))


def test_ance_requires_two_visible_gpus():
    from train_ance import require_ance_gpus
    assert require_ance_gpus(2) == 2
    assert require_ance_gpus(4) == 4
    for count in (0, 1):
        _assert_raises(RuntimeError, lambda n=count: require_ance_gpus(n), "2 visible GPUs")


def test_inferencer_help_exits_cleanly():
    result = subprocess.run(
        [sys.executable, str(project_root / 'scripts' / 'run_ance_data_gen.py'), '--help'],
        text=True, capture_output=True, check=False)
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert 'Traceback' not in result.stderr


TESTS = [
    ("commit: metadata first, marker last", test_marker_is_written_last_and_metadata_first),
    ("commit: no marker => invisible round", test_a_round_without_its_marker_is_invisible),
    ("commit: marker without metadata refused", test_marker_without_metadata_is_refused),
    ("commit: foreign run's round refused", test_a_foreign_runs_round_is_refused),
    ("commit: truncated round data refused", test_truncated_round_data_is_refused),
    ("commit: same-count mutation refused", test_same_count_round_mutation_is_refused),
    ("commit: extra file in a round refused", test_extra_file_in_a_round_is_refused),
    ("commit: missing data directory refused", test_missing_data_directory_is_refused),
    ("commit: sampling failures block publication", test_sampling_failures_block_publication_entirely),
    ("commit: ready_initial is not a numeric round", test_initial_round_is_not_counted_as_a_numeric_round),
    ("commit: latest tracks the highest marker", test_latest_committed_round_tracks_the_highest_marker),
    ("fresh: initial round alone fails the gate", test_initial_round_alone_does_not_satisfy_the_gate),
    ("fresh: no rounds fails the gate", test_no_rounds_at_all_does_not_satisfy_the_gate),
    ("fresh: unconsumed round does not count", test_a_published_but_never_consumed_round_does_not_count),
    ("fresh: one consumed round passes", test_one_consumed_checkpoint_round_satisfies_the_gate),
    ("fresh: min_consume_steps enforced", test_min_consume_steps_is_enforced),
    ("fresh: min_fresh_rounds enforced", test_min_fresh_rounds_is_enforced),
    ("supervise: dead inferencer kills the trainer", test_a_dead_inferencer_kills_the_trainer_and_fails_the_run),
    ("supervise: simultaneous death still caught", test_an_inferencer_that_dies_with_the_trainer_is_still_caught),
    ("supervise: healthy run reports no failure", test_a_healthy_run_reports_no_failure),
    ("supervise: clean EARLY exit is a failure", test_a_clean_early_inferencer_exit_is_a_failure),
    ("supervise: our own termination is not a failure", test_our_own_termination_is_not_an_early_exit),
    ("loss: non-finite rejected before backward", test_non_finite_loss_is_rejected_before_backward),
    ("grad: non-finite norm rejected", test_non_finite_grad_norm_is_rejected),
    ("grad: no step taken on non-finite gradients", test_apply_gradients_does_not_step_on_non_finite_gradients),
    ("grad: finite gradients do step", test_apply_gradients_steps_on_finite_gradients),
    ("grad: finite loss does not imply finite gradients", test_a_finite_loss_does_not_imply_finite_gradients),
    ("summary: retried then raises", test_summary_write_is_retried_then_raises),
    ("summary: survives a transient failure", test_summary_write_survives_a_transient_failure),
    ("summary: a failed write is fatal", test_trainer_treats_a_failed_summary_write_as_fatal),
    ("runid: unique within one second", test_run_ids_are_unique_within_one_second),
    ("runid: carries fingerprint and job id", test_run_id_carries_the_fingerprint_and_job_id),
    ("runid: omits job id outside SLURM", test_run_id_omits_the_job_id_outside_slurm),
    ("runid: existing work root refused", test_work_root_creation_refuses_an_existing_directory),
    ("gpu: two visible devices required", test_ance_requires_two_visible_gpus),
    ("cli: inferencer help exits cleanly", test_inferencer_help_exits_cleanly),
    ("success: terminal record validates a complete run", test_a_complete_run_validates_via_the_terminal_record),
    ("success: no terminal record => rejected", test_without_the_terminal_record_a_complete_run_is_rejected),
    ("success: zero-step run rejected", test_a_zero_step_run_is_rejected),
    ("success: non-finite loss in the log rejected", test_a_non_finite_loss_in_the_log_is_rejected),
    ("success: manifest records run identity", test_manifest_records_the_run_identity_for_the_evaluator),
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
    print("\nANCE pipeline tests (handoff, supervision, freshness, success)")
    print("=" * 66)
    passed = sum(_run(n, f) for n, f in TESTS)
    print("=" * 66)
    print(f"  {passed}/{len(TESTS)} passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
