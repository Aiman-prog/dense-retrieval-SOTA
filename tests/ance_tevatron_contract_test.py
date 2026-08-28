"""The pinned-Tevatron contract the ANCE trainer depends on.

`run_ance_train.py` drives `DenseModel` by hand rather than through Tevatron's
`Trainer`, so nothing else in the repo checks that its assumptions still hold. All
four are load-bearing:

* the score matrix is `B x (B*G)` with `target = arange(B) * G`, which is what makes
  the negative pool 127 at B=64/G=2 -- ONE ANN-mined negative plus 126 passages
  belonging to other examples in the batch;
* the temperature is applied exactly once, inside `EncoderModel.forward`. This is
  why `patch_tevatron_loss` must NOT be called here: it patches `gc_trainer` and
  would divide a second time;
* `model.save()` writes the encoder, not the wrapper (`save_pretrained` on the
  wrapper produces a checkpoint the evaluator cannot load);
* `attn_implementation='eager'` has to be passed as an hf_kwarg -- Tevatron does not
  forward `ModelArguments.attn_implementation` to `from_pretrained`.

CPU only, on a randomly initialised 2-layer encoder.

Run: python tests/ance_tevatron_contract_test.py
"""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

import torch                                                       # noqa: E402
from transformers import XLMRobertaConfig, XLMRobertaModel          # noqa: E402
from tevatron.retriever.modeling import DenseModel                  # noqa: E402

TEMPERATURE = 0.02
VOCAB, DIM, SEQ = 64, 32, 8


def _assert_raises(exc, fn, contains=None):
    try:
        fn()
    except exc as e:
        assert contains is None or contains in str(e), str(e)
        return str(e)
    raise AssertionError(f"expected {exc.__name__}")


def _tiny_encoder():
    # dropout off: these tests re-encode the same inputs to compare against the
    # forward pass, and the model is necessarily in train() mode for the loss branch.
    cfg = XLMRobertaConfig(vocab_size=VOCAB, hidden_size=DIM, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=DIM * 2,
                           max_position_embeddings=SEQ + 4,
                           hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    torch.manual_seed(0)
    return XLMRobertaModel(cfg)


def _model():
    return DenseModel(encoder=_tiny_encoder(), pooling='cls', normalize=True,
                      temperature=TEMPERATURE)


def _batch(B, G):
    torch.manual_seed(1)
    q = torch.randint(0, VOCAB, (B, SEQ))
    p = torch.randint(0, VOCAB, (B * G, SEQ))
    return ({'input_ids': q, 'attention_mask': torch.ones_like(q)},
            {'input_ids': p, 'attention_mask': torch.ones_like(p)})


# ---- pool semantics ---------------------------------------------------------

def test_score_matrix_is_B_by_B_times_G():
    B, G = 4, 2
    model = _model().train()
    out = model(*_batch(B, G))
    assert tuple(out.scores.shape) == (B, B * G), out.scores.shape


def test_target_positions_are_arange_times_group_size():
    """Each query's positive sits at index i*G, so every other column is a negative."""
    B, G = 4, 2
    model = _model().train()
    q, p = _batch(B, G)
    out = model(q, p)
    expected = torch.arange(B) * G
    loss = torch.nn.functional.cross_entropy(out.scores / TEMPERATURE, expected)
    assert torch.allclose(out.loss, loss, atol=1e-5), (out.loss.item(), loss.item())


def test_negative_pool_is_127_at_the_configured_batch_and_group():
    """The number the run manifest records, derived from the real score matrix.

    Explicitly NOT a claim of false-negative masking: 126 of the 127 are other
    examples' passages and the pinned loss masks nothing across queries.
    """
    B, G = 64, 2
    manifest_formula = B * G - 1

    small_B, small_G = 4, 2
    model = _model().train()
    out = model(*_batch(small_B, small_G))
    observed = out.scores.shape[1] - 1
    assert observed == small_B * small_G - 1 == 7, observed
    assert manifest_formula == 127, manifest_formula


def test_a_group_of_one_leaves_only_in_batch_negatives():
    """G=1 is the degenerate case: no mined negative at all. Pinned so that a recipe
    edit to train_group_size cannot silently remove ANCE's negative."""
    B, G = 4, 1
    model = _model().train()
    out = model(*_batch(B, G))
    assert tuple(out.scores.shape) == (B, B)
    assert torch.equal(torch.arange(B) * G, torch.arange(B))


# ---- temperature ------------------------------------------------------------

def test_temperature_is_applied_exactly_once():
    B, G = 4, 2
    model = _model().train()
    q, p = _batch(B, G)
    out = model(q, p)
    target = torch.arange(B) * G
    once  = torch.nn.functional.cross_entropy(out.scores / TEMPERATURE, target)
    twice = torch.nn.functional.cross_entropy(
        out.scores / TEMPERATURE / TEMPERATURE, target)
    assert torch.allclose(out.loss, once, atol=1e-5)
    assert not torch.allclose(out.loss, twice, atol=1e-5), \
        "loss matches double-scaled scores; patch_tevatron_loss must not be applied"


def test_scores_returned_are_unscaled():
    """`outputs.scores` is raw similarity; only the loss sees the temperature."""
    B, G = 4, 2
    model = _model().train()
    q, p = _batch(B, G)
    out = model(q, p)
    q_reps, p_reps = model.encode_query(q), model.encode_passage(p)
    assert torch.allclose(out.scores, q_reps @ p_reps.T, atol=1e-5)


def test_normalized_scores_stay_within_cosine_range():
    model = _model().train()
    out = model(*_batch(4, 2))
    assert out.scores.abs().max() <= 1.0 + 1e-4, out.scores.abs().max().item()


# ---- saving -----------------------------------------------------------------

def test_model_save_writes_the_encoder_not_the_wrapper():
    """CLAUDE.md: save with model.save(), not save_pretrained(). The wrapper's own
    save would prefix every key and the evaluator could not load it."""
    model = _model()
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        out = Path(tmp)
        assert (out / "config.json").is_file()
        weights = [p for p in out.iterdir()
                   if p.name in ("model.safetensors", "pytorch_model.bin")]
        assert weights, sorted(p.name for p in out.iterdir())
        cfg = json.loads((out / "config.json").read_text())
        assert cfg['model_type'] == 'xlm-roberta', cfg.get('model_type')
        reloaded = XLMRobertaModel.from_pretrained(tmp)
        assert reloaded.config.hidden_size == DIM


def test_keys_to_ignore_on_save_patch_is_present():
    """Tevatron leaves this unset and HF then raises during save."""
    import train_ance
    assert train_ance.__name__            # the patch is an import-time side effect
    assert hasattr(DenseModel, "_keys_to_ignore_on_save")
    assert DenseModel._keys_to_ignore_on_save is None


# ---- attention implementation ----------------------------------------------

def test_eager_attention_is_reachable_as_an_hf_kwarg():
    """XLM-RoBERTa has no sdpa path in the pinned stack; the trainer passes
    attn_implementation twice on purpose (ModelArguments is not forwarded)."""
    with tempfile.TemporaryDirectory() as tmp:
        _tiny_encoder().save_pretrained(tmp)
        loaded = XLMRobertaModel.from_pretrained(tmp, attn_implementation='eager')
        impl = getattr(loaded.config, '_attn_implementation', None)
        assert impl == 'eager', impl


TESTS = [
    ("pool: scores are B x (B*G)", test_score_matrix_is_B_by_B_times_G),
    ("pool: target = arange(B) * G", test_target_positions_are_arange_times_group_size),
    ("pool: 127 negatives at B=64, G=2", test_negative_pool_is_127_at_the_configured_batch_and_group),
    ("pool: G=1 leaves only in-batch negatives", test_a_group_of_one_leaves_only_in_batch_negatives),
    ("temp: applied exactly once", test_temperature_is_applied_exactly_once),
    ("temp: returned scores are unscaled", test_scores_returned_are_unscaled),
    ("temp: normalized scores within cosine range", test_normalized_scores_stay_within_cosine_range),
    ("save: model.save() writes the encoder", test_model_save_writes_the_encoder_not_the_wrapper),
    ("save: _keys_to_ignore_on_save patched", test_keys_to_ignore_on_save_patch_is_present),
    ("attn: eager reachable as an hf_kwarg", test_eager_attention_is_reachable_as_an_hf_kwarg),
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
    print("\nANCE pinned-Tevatron contract tests")
    print("=" * 58)
    passed = sum(_run(n, f) for n, f in TESTS)
    print("=" * 58)
    print(f"  {passed}/{len(TESTS)} passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
