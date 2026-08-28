"""Paper-fidelity ANCE: reference parity, and proof the BGE arm is untouched.

Every expected value here is derived from `microsoft/ANCE` (checked out locally at
/Users/aiamn/PycharmProjects/ANCE), NOT from our implementation, so these tests can
fail our port rather than merely describe it:

    architecture  model/models.py:137-157   roberta -> Linear(h,768) -> LayerNorm, CLS
    loss          model/models.py:77-81     -log_softmax([q.pos, q.neg], 1)[:, 0]
    LAMB          utils/lamb.py             no debiasing; weight_norm clamp (0,10);
                                            wd inside adam_step; trust_ratio 1 on zero
    mining        run_ann_data_gen.py:366   shuffled top-200, 20 non-positive
    consumption   msmarco_data.py:337-362   one triplet per negative

The other half of this file is the negative space: the BRIGHT ANCE and GRASS arms share
one objective so the BRIGHT table isolates negative SELECTION, and nothing here may
change that.

Run: python tests/ance_paper_test.py
"""
import contextlib
import json
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

import torch                                                       # noqa: E402
import torch.nn.functional as F                                    # noqa: E402
from transformers import RobertaConfig                             # noqa: E402

from ance_paper import (                                           # noqa: E402
    ALLOWED_UNEXPECTED_PREFIXES, AnceEncoder, EMBED_DIM, Lamb,
    encode_jsonl_to_pickle, load_ance_encoder, pairwise_nll,
)
from ance_mining import select_ance_negatives                      # noqa: E402
from utils.helpers import build_faiss_index                        # noqa: E402

VOCAB, HIDDEN, SEQ = 64, 32, 8


def _assert_raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        assert contains is None or contains in str(e), str(e)
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


def _tiny_config():
    return RobertaConfig(vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=HIDDEN * 2,
                         max_position_embeddings=SEQ + 4,
                         hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)


def _tiny(seed=0):
    torch.manual_seed(seed)
    return AnceEncoder(_tiny_config())


def _inputs(n=3, seed=1):
    torch.manual_seed(seed)
    ids = torch.randint(0, VOCAB, (n, SEQ))
    return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


def _upstream_state(model, *, with_classifier=True, drop=(), extra=None):
    """A state dict shaped like a released checkpoint.

    transformers 2.3.0 wrote these through RobertaForSequenceClassification, so a real
    one carries `classifier.*` alongside the encoder and the head.
    """
    state = dict(model.state_dict())
    if with_classifier:
        state["classifier.dense.weight"] = torch.zeros(HIDDEN, HIDDEN)
        state["classifier.dense.bias"] = torch.zeros(HIDDEN)
        state["classifier.out_proj.weight"] = torch.zeros(2, HIDDEN)
        state["classifier.out_proj.bias"] = torch.zeros(2)
    for key in drop:
        state.pop(key, None)
    state.update(extra or {})
    return state


def _write_checkpoint(directory, state, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(str(directory))
    torch.save(state, directory / "pytorch_model.bin")
    return directory


# ---- 1. loading a released checkpoint ---------------------------------------

def test_an_upstream_checkpoint_loads_with_the_head_intact():
    """The whole reason no importer is needed: their key names ARE our attribute
    names, so from_pretrained maps roberta.*/embeddingHead.*/norm.* directly."""
    with tempfile.TemporaryDirectory() as tmp:
        model = _tiny().eval()
        d = _write_checkpoint(Path(tmp) / "released", _upstream_state(model),
                              model.config)
        loaded = load_ance_encoder(d).eval()
        batch = _inputs()
        with torch.no_grad():
            assert torch.allclose(model(**batch), loaded(**batch), rtol=0, atol=1e-6)
        # the head really came from the file, not from a fresh init
        assert torch.allclose(loaded.embeddingHead.weight, model.embeddingHead.weight)
        assert torch.allclose(loaded.norm.bias, model.norm.bias)


def test_the_unused_classifier_head_is_tolerated():
    """Rejecting it would make the importer-free path unable to read the real
    checkpoints, which all carry one."""
    with tempfile.TemporaryDirectory() as tmp:
        model = _tiny()
        d = _write_checkpoint(Path(tmp) / "c", _upstream_state(model, with_classifier=True),
                              model.config)
        assert load_ance_encoder(d) is not None


def test_a_missing_head_key_is_refused():
    """from_pretrained only WARNS on missing keys. A warning in a 24-hour job's log is
    not a gate, and a randomly initialized projection head trains and converges to
    something that is not ANCE."""
    with tempfile.TemporaryDirectory() as tmp:
        model = _tiny()
        d = _write_checkpoint(Path(tmp) / "d",
                              _upstream_state(model, drop=("norm.bias",)), model.config)
        _assert_raises(ValueError, lambda: load_ance_encoder(d), "norm.bias")


def test_an_unknown_extra_key_is_refused():
    """The tolerance is an ALLOWLIST. A key nobody understands could be a head this
    port does not know about."""
    with tempfile.TemporaryDirectory() as tmp:
        model = _tiny()
        d = _write_checkpoint(Path(tmp) / "e",
                              _upstream_state(model, extra={"adapter.weight":
                                                            torch.zeros(4)}),
                              model.config)
        _assert_raises(ValueError, lambda: load_ance_encoder(d), "adapter.weight")


def test_the_allowlist_stays_narrow():
    """A widened prefix ('' or 'c') would swallow real keys silently."""
    assert ALLOWED_UNEXPECTED_PREFIXES == ("classifier.",), ALLOWED_UNEXPECTED_PREFIXES


def test_save_pretrained_round_trips_the_head():
    """The head lives inside the weight file, so no sidecar and no cache-identity
    special case are needed."""
    with tempfile.TemporaryDirectory() as tmp:
        model = _tiny().eval()
        with torch.no_grad():
            model.embeddingHead.weight.fill_(0.123)
            model.norm.bias.fill_(0.456)
        model.save_pretrained(tmp)
        back = load_ance_encoder(tmp).eval()
        batch = _inputs()
        with torch.no_grad():
            assert torch.allclose(model(**batch), back(**batch), rtol=0, atol=1e-6)


def test_the_architecture_matches_the_reference():
    model = _tiny().eval()
    assert model.embeddingHead.in_features == HIDDEN
    assert model.embeddingHead.out_features == EMBED_DIM == 768
    assert model.norm.normalized_shape == (768,)
    batch = _inputs()
    with torch.no_grad():
        hidden = model.roberta(**batch).last_hidden_state
        expected = model.norm(model.embeddingHead(hidden[:, 0]))   # CLS, by hand
        assert torch.allclose(model(**batch), expected, atol=1e-6)


def test_embeddings_are_not_l2_normalized():
    model = _tiny().eval()
    with torch.no_grad():
        norms = model(**_inputs(8)).norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-2), \
        "embeddings look L2-normalized; the port has drifted to cosine similarity"


# ---- 2. the loss ------------------------------------------------------------

def test_pairwise_nll_matches_the_reference_formula():
    torch.manual_seed(3)
    q, p, n = (torch.randn(5, EMBED_DIM) for _ in range(3))
    logits = torch.cat([(q * p).sum(-1, keepdim=True),
                        (q * n).sum(-1, keepdim=True)], dim=1)
    expected = (-F.log_softmax(logits, dim=1)[:, 0]).mean()
    assert torch.allclose(pairwise_nll(q, p, n), expected, atol=1e-6)


def test_the_loss_uses_dot_product_not_cosine():
    """Scaling the positive changes a dot-product loss; a cosine loss ignores it."""
    torch.manual_seed(4)
    q, p, n = (torch.randn(4, EMBED_DIM) for _ in range(3))
    assert not torch.allclose(pairwise_nll(q, p, n), pairwise_nll(q, p * 3.0, n),
                              atol=1e-4), "loss is scale-invariant, i.e. cosine"


def test_the_loss_applies_no_temperature():
    torch.manual_seed(5)
    q, p, n = (torch.randn(4, EMBED_DIM) for _ in range(3))
    logits = torch.cat([(q * p).sum(-1, keepdim=True),
                        (q * n).sum(-1, keepdim=True)], dim=1)
    tempered = (-F.log_softmax(logits / 0.02, dim=1)[:, 0]).mean()
    assert not torch.allclose(pairwise_nll(q, p, n), tempered, atol=1e-4)


# ---- 3. LAMB ----------------------------------------------------------------

def _reference_lamb_step(p, grad, lr, betas, eps, weight_decay):
    """utils/lamb.py, transcribed. No debiasing, clamp, wd inside adam_step."""
    beta1, beta2 = betas
    exp_avg = (1 - beta1) * grad
    exp_avg_sq = (1 - beta2) * grad * grad
    weight_norm = p.pow(2).sum().sqrt().clamp(0, 10)
    adam_step = exp_avg / (exp_avg_sq.sqrt() + eps)
    if weight_decay != 0:
        adam_step = adam_step + weight_decay * p
    adam_norm = adam_step.pow(2).sum().sqrt()
    trust_ratio = 1 if weight_norm == 0 or adam_norm == 0 else weight_norm / adam_norm
    return p - lr * trust_ratio * adam_step


def _one_lamb_step(weight_decay=0.5, lr=0.1, eps=1e-8, adam=False):
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.randn(6))
    grad = torch.randn(6)
    before = p.data.clone()
    opt = Lamb([p], lr=lr, eps=eps, weight_decay=weight_decay, adam=adam)
    p.grad = grad.clone()
    opt.step()
    return before, grad, p.data.clone(), opt.state[p]


def test_lamb_update_matches_the_reference():
    before, grad, after, _ = _one_lamb_step()
    expected = _reference_lamb_step(before, grad, 0.1, (0.9, 0.999), 1e-8, 0.5)
    assert torch.allclose(after, expected, rtol=0, atol=1e-7), (after - expected)


def test_lamb_does_not_debias():
    """Upstream: "Paper v3 does not use debiasing" -- step_size is the raw lr.

    weight_decay MUST be non-zero here: at wd=0 the trust ratio exactly cancels the
    debiasing scale factor, so step 1 cannot tell the two apart and the test would be
    vacuous.
    """
    before, grad, after, _ = _one_lamb_step(weight_decay=0.5)
    plain = _reference_lamb_step(before, grad, 0.1, (0.9, 0.999), 1e-8, 0.5)
    bias_c1, bias_c2 = 1 - 0.9, 1 - 0.999
    debiased_lr = 0.1 * (bias_c2 ** 0.5) / bias_c1
    debiased = _reference_lamb_step(before, grad, debiased_lr, (0.9, 0.999), 1e-8, 0.5)
    assert not torch.allclose(plain, debiased, rtol=0, atol=1e-7), \
        "the test cannot discriminate; pick a configuration where debiasing matters"
    assert torch.allclose(after, plain, rtol=0, atol=1e-7)


def test_lamb_clamps_the_weight_norm_at_ten():
    torch.manual_seed(8)
    p = torch.nn.Parameter(torch.full((4,), 50.0))     # norm 100, far above the clamp
    opt = Lamb([p], lr=0.1, eps=1e-8, weight_decay=0.0)
    p.grad = torch.randn(4)
    opt.step()
    assert float(opt.state[p]['weight_norm']) == 10.0, opt.state[p]['weight_norm']


def test_lamb_trust_ratio_is_one_on_a_zero_norm():
    p = torch.nn.Parameter(torch.zeros(4))             # weight_norm == 0
    opt = Lamb([p], lr=0.1, eps=1e-8, weight_decay=0.0)
    p.grad = torch.randn(4)
    opt.step()
    assert opt.state[p]['trust_ratio'] == 1


def test_lamb_weight_decay_is_inside_adam_step():
    """Folded in BEFORE the trust ratio scales it. Applied after, the update differs."""
    before, grad, after, _ = _one_lamb_step(weight_decay=0.5)
    inside = _reference_lamb_step(before, grad, 0.1, (0.9, 0.999), 1e-8, 0.5)
    outside = _reference_lamb_step(before, grad, 0.1, (0.9, 0.999), 1e-8, 0.0) \
        - 0.1 * 0.5 * before
    assert not torch.allclose(inside, outside, rtol=0, atol=1e-7)
    assert torch.allclose(after, inside, rtol=0, atol=1e-7)


# ---- 4. mining and consumption ----------------------------------------------

def test_twenty_negatives_from_the_shuffled_top_200():
    """run_ann_data_gen.py:366-389 -- shuffle the full top-k, take the first 20
    non-positive, non-duplicate. Our select_ance_negatives is the same procedure at
    n_negs=20; the BRIGHT arm just calls it with n_negs=1."""
    candidates = [str(i) for i in range(200)]
    negs = select_ance_negatives("q1", candidates, {"7", "8"}, n_negs=20,
                                 rng=random.Random(0))
    assert len(negs) == 20 and len(set(negs)) == 20
    assert not ({"7", "8"} & set(negs))
    assert set(negs) <= set(candidates)


def test_negative_selection_is_seeded():
    candidates = [str(i) for i in range(200)]
    a = select_ance_negatives("q", candidates, set(), n_negs=20, rng=random.Random(3))
    b = select_ance_negatives("q", candidates, set(), n_negs=20, rng=random.Random(3))
    assert a == b


def test_twenty_negatives_become_twenty_triplets():
    """--triplet / msmarco_data.py:337-362: each negative is its own instance, so each
    loss term still sees exactly one positive and one negative."""
    from run_ance_train import ANCEDataset

    with tempfile.TemporaryDirectory() as tmp:
        with open(Path(tmp) / "r.jsonl", 'w') as f:
            for q in range(3):
                f.write(json.dumps({
                    'query_id': str(q), 'query': f'q{q}',
                    'positive_passages': [{'docid': 'p', 'text': f'pos{q}'}],
                    'negative_passages': [{'docid': f'n{i}', 'text': f'neg{q}_{i}'}
                                          for i in range(20)]}) + '\n')
        ds = ANCEDataset(tmp, None, SEQ, SEQ, 21, paper_mode=True)
        assert len(ds) == 60, len(ds)
        query, pos, neg = ds[0]
        assert (query, pos) == ('q0', 'pos0') and neg.startswith('neg0_')
        assert len({t[2] for t in ds.triplets[:20]}) == 20


# ---- 5. the miner seam ------------------------------------------------------

def test_the_paper_encoder_emits_the_faiss_pickle_contract():
    """Emitting the SAME (embeddings, ids) tuple is what leaves build_faiss_index,
    mine_from_index, publish_round and read_round untouched -- and why
    run_ance_data_gen.py needs no paper branch at all."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = _tiny()
        model.save_pretrained(tmp / "m")

        corpus = tmp / "corpus.jsonl"
        with open(corpus, 'w') as f:
            for i in range(5):
                f.write(json.dumps({'docid': f'd{i}', 'text': f'passage {i}'}) + '\n')

        with _fake_tokenizer():
            out = encode_jsonl_to_pickle(tmp / "m", corpus, tmp / "c.pkl",
                                         is_query=False, max_len=SEQ, batch_size=2)
        index, embeddings, ids = build_faiss_index(out)
        assert ids == [f'd{i}' for i in range(5)]
        assert embeddings.shape == (5, EMBED_DIM)
        assert index.ntotal == 5


@contextlib.contextmanager
def _fake_tokenizer():
    """A real RoBERTa tokenizer is not what the seam test is about. Restored on exit:
    a leaked patch would silently change every later test in this process."""
    import transformers
    real = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = staticmethod(
        lambda *a, **k: _FakeTokenizer())
    try:
        yield
    finally:
        transformers.AutoTokenizer.from_pretrained = real


class _FakeTokenizer:
    def __call__(self, texts, **kw):
        texts = list(texts)
        ids = torch.tensor([[(sum(map(ord, t)) + i) % VOCAB for i in range(SEQ)]
                            for t in texts])
        return _Batch({'input_ids': ids, 'attention_mask': torch.ones_like(ids)})


class _Batch(dict):
    def to(self, device):
        return _Batch({k: v.to(device) for k, v in self.items()})


# ---- 6. the BGE arm is untouched --------------------------------------------

def test_existing_recipes_keep_the_shared_objective():
    """BRIGHT ANCE and GRASS share one objective so the BRIGHT table isolates negative
    SELECTION. A recipe drifting off it silently re-opens that comparison."""
    from utils.helpers import load_config, effective_model_config
    config = load_config()
    baseline = dict(config['model'])
    for name, recipe in config['training'].items():
        if not isinstance(recipe, dict) or recipe.get('paper_fidelity'):
            continue
        assert effective_model_config(config, recipe) == baseline, name
        assert baseline['normalize'] is True and float(baseline['temperature']) == 0.02


def test_the_paper_recipe_declares_the_paper_objective():
    """normalize:false alone is not enough -- the manifest and the embedding cache
    would still inherit temperature 0.02 and describe a scaled objective."""
    from utils.helpers import load_config, effective_model_config
    config = load_config()
    recipe = config['training']['ance_paper']
    model = effective_model_config(config, recipe)
    assert model['normalize'] is False and float(model['temperature']) == 1.0
    assert model['query_max_len'] == 64 and model['passage_max_len'] == 512
    # the upstream passage values, run_train.sh:24-36 and :93
    assert float(recipe['learning_rate']) == 1e-6
    assert float(recipe['lamb_eps']) == 1e-8
    assert float(recipe['weight_decay']) == 0.0
    assert int(recipe['warmup_steps']) == 5000
    assert int(recipe['mining_depth']) == 200
    assert int(recipe['train_group_size']) - 1 == 20
    assert int(recipe['total_epochs']) == 2
    assert int(recipe['save_steps']) == 10000


def test_paper_budget_is_two_epochs_over_all_twenty_triplets():
    """The MS MARCO export has one row/query, but paper mode expands each row into
    twenty triplets. Budgeting from rows would silently train on only 1/20 of an epoch."""
    from train_ance import calculate_training_budget

    recipe = {'batch_size': 64, 'total_epochs': 2, 'train_group_size': 21,
              'paper_fidelity': True}
    budget = calculate_training_budget(400_782, recipe)
    assert budget == {
        'query_records': 400_782,
        'triplets_per_query': 20,
        'training_instances': 8_015_640,
        'steps_per_epoch': 125_244,
        'total_epochs': 2,
        'max_steps': 250_488,
        'triplets_processed': 16_031_232,
    }, budget


def test_nonpaper_budget_is_not_multiplied_by_group_size():
    """The BGE loader consumes one grouped record per item; only pairwise paper mode
    expands negatives into separate dataset items."""
    from train_ance import calculate_training_budget

    budget = calculate_training_budget(
        10, {'batch_size': 4, 'total_epochs': 2, 'train_group_size': 2})
    assert budget['triplets_per_query'] == 1
    assert budget['training_instances'] == 10
    assert budget['steps_per_epoch'] == 2
    assert budget['max_steps'] == 4


def test_paper_manifest_distinguishes_candidates_from_loss_pool():
    """Twenty mined candidates become separate pairwise examples; the loss never sees
    the 1,343-way pool implied by the BGE in-batch formula 64*21-1."""
    from train_ance import negative_pool_manifest

    paper = negative_pool_manifest(
        {'paper_fidelity': True, 'train_group_size': 21}, batch_size=64)
    assert paper == {'negative_pool_size': 1, 'mined_negatives_per_query': 20,
                     'triplets_per_query': 20}
    bge = negative_pool_manifest(
        {'train_group_size': 2}, batch_size=64)
    assert bge == {'negative_pool_size': 127, 'mined_negatives_per_query': 1,
                   'triplets_per_query': 1}


def test_paper_work_root_and_launchers_are_reachable():
    from utils.helpers import get_path

    assert get_path('temp_ance_paper').name == 'temp_ance_paper_workdir'
    train_launcher = (project_root / 'scripts' / 'launchers' /
                      'run_ance_msmarco_singularity.sh').read_text()
    eval_launcher = (project_root / 'scripts' / 'launchers' /
                     'eval_msmarco_singularity.sh').read_text()
    assert 'ANCE_RECIPE' in train_launcher and '--recipe "${ANCE_RECIPE}"' in train_launcher
    assert 'EVAL_RECIPE' in eval_launcher and '--recipe "${EVAL_RECIPE}"' in eval_launcher
    assert 'EVAL_MODEL_PATH' in eval_launcher


def test_reproduction_datasets_have_independent_immutable_revisions():
    import yaml

    config = yaml.safe_load((project_root / 'config' / 'config.yaml').read_text())
    revisions = config['data']['msmarco_reproduction']
    assert set(revisions) == {'passage_revision', 'corpus_revision'}
    assert all(len(value) == 40 and all(c in '0123456789abcdef' for c in value)
               for value in revisions.values())
    assert revisions['passage_revision'] != revisions['corpus_revision']


def test_post_training_command_preserves_the_selected_recipe():
    import contextlib
    import io

    from train_ance import _print_eval_instructions

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        _print_eval_instructions(
            'ance_paper', {'eval_corpus_file': 'corpus.jsonl'}, '/model')
    output = output.getvalue()
    assert 'eval_msmarco.py --recipe ance_paper' in output


def test_only_the_paper_arm_leaves_the_tevatron_encode_route():
    """encode_to_pickle is the single seam: 13 call sites route through it. The BGE
    arms must still shell out to Tevatron's driver."""
    import utils.helpers as helpers
    calls = {}
    real_run = helpers.subprocess.run
    helpers.subprocess.run = lambda cmd, **k: calls.setdefault('cmd', cmd)
    try:
        ctx = {'args': {'per_device_eval_batch_size': 8, 'dataloader_num_workers': 0},
               'pooling': 'cls', 'normalize': True, 'max_q': 32, 'max_p': 64}
        helpers.encode_to_pickle('/model', '/in.jsonl', Path('/tmp/x/o.pkl'), True,
                                 ctx, {'model': {}})
    finally:
        helpers.subprocess.run = real_run
    assert 'tevatron.retriever.driver.encode' in calls['cmd'], calls['cmd']


# ---- 7. the preflight verdict -----------------------------------------------

def test_the_reproduction_verdict_boundaries():
    """The preflight's whole output. A tolerance chosen after seeing the number is not
    a criterion, so the bar is pinned here."""
    from eval_msmarco import (reproduction_verdict, PAPER_MRR_AT_10,
                              PAPER_RECALL_AT_1000)
    exact, deltas = reproduction_verdict(PAPER_MRR_AT_10, PAPER_RECALL_AT_1000, True)
    assert exact is True and deltas == {'mrr_at_10': 0.0, 'recall_1000': 0.0}
    edge, _ = reproduction_verdict(PAPER_MRR_AT_10 - 0.005,
                                   PAPER_RECALL_AT_1000 + 0.005, True)
    assert edge is True, "exactly the tolerance must pass"
    over, _ = reproduction_verdict(PAPER_MRR_AT_10 - 0.0051, PAPER_RECALL_AT_1000, True)
    assert over is False
    # one metric inside, the other outside, is still a failure
    half, _ = reproduction_verdict(PAPER_MRR_AT_10, PAPER_RECALL_AT_1000 - 0.02, True)
    assert half is False


def test_no_verdict_off_the_official_dev_split():
    """MRR@10 on a different denominator compares to nothing, so it gets no verdict."""
    from eval_msmarco import (reproduction_verdict, msmarco_paper_comparable,
                              PAPER_MRR_AT_10, PAPER_RECALL_AT_1000)
    assert msmarco_paper_comparable(6980) and not msmarco_paper_comparable(6979)
    verdict, _ = reproduction_verdict(PAPER_MRR_AT_10, PAPER_RECALL_AT_1000, False)
    assert verdict is None


TESTS = [
    ("load: upstream checkpoint, head intact", test_an_upstream_checkpoint_loads_with_the_head_intact),
    ("load: unused classifier tolerated", test_the_unused_classifier_head_is_tolerated),
    ("load: missing head key refused", test_a_missing_head_key_is_refused),
    ("load: unknown extra key refused", test_an_unknown_extra_key_is_refused),
    ("load: allowlist stays narrow", test_the_allowlist_stays_narrow),
    ("load: save_pretrained round-trips the head", test_save_pretrained_round_trips_the_head),
    ("arch: matches the reference", test_the_architecture_matches_the_reference),
    ("arch: embeddings are not normalized", test_embeddings_are_not_l2_normalized),
    ("loss: matches the reference formula", test_pairwise_nll_matches_the_reference_formula),
    ("loss: dot product, not cosine", test_the_loss_uses_dot_product_not_cosine),
    ("loss: no temperature", test_the_loss_applies_no_temperature),
    ("lamb: update matches the reference", test_lamb_update_matches_the_reference),
    ("lamb: no debiasing", test_lamb_does_not_debias),
    ("lamb: weight_norm clamped at 10", test_lamb_clamps_the_weight_norm_at_ten),
    ("lamb: trust_ratio 1 on a zero norm", test_lamb_trust_ratio_is_one_on_a_zero_norm),
    ("lamb: weight decay inside adam_step", test_lamb_weight_decay_is_inside_adam_step),
    ("mine: 20 from the shuffled top-200", test_twenty_negatives_from_the_shuffled_top_200),
    ("mine: seeded and reproducible", test_negative_selection_is_seeded),
    ("mine: 20 negatives -> 20 triplets", test_twenty_negatives_become_twenty_triplets),
    ("seam: encoder emits the pickle contract", test_the_paper_encoder_emits_the_faiss_pickle_contract),
    ("verdict: +/-0.005 boundaries", test_the_reproduction_verdict_boundaries),
    ("verdict: none off the official split", test_no_verdict_off_the_official_dev_split),
    ("bge: existing recipes keep the objective", test_existing_recipes_keep_the_shared_objective),
    ("bge: paper recipe declares its own", test_the_paper_recipe_declares_the_paper_objective),
    ("budget: two epochs over twenty triplets", test_paper_budget_is_two_epochs_over_all_twenty_triplets),
    ("budget: BGE records are not expanded", test_nonpaper_budget_is_not_multiplied_by_group_size),
    ("manifest: candidates differ from loss pool", test_paper_manifest_distinguishes_candidates_from_loss_pool),
    ("launch: paper work root and env dispatch", test_paper_work_root_and_launchers_are_reachable),
    ("data: independent immutable revisions", test_reproduction_datasets_have_independent_immutable_revisions),
    ("eval: selected recipe survives handoff", test_post_training_command_preserves_the_selected_recipe),
    ("bge: only the paper arm leaves Tevatron", test_only_the_paper_arm_leaves_the_tevatron_encode_route),
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
    print("\nPaper-fidelity ANCE — reference parity")
    print("=" * 58)
    passed = sum(_run(n, f) for n, f in TESTS)
    print("=" * 58)
    print(f"  {passed}/{len(TESTS)} passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
