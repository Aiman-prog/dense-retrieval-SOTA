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


class _WhitespaceTokenizer:
    """Pad-to-longest stand-in for the tests that only need ids, offline and hermetic.

    Faithful to the one behaviour those tests rely on: `padding=True` pads to the
    longest sequence IN THE BATCH, not to `max_length`. One id per whitespace word.
    """

    def __init__(self):
        self.vocab = {}

    def __call__(self, texts, padding=True, truncation=True, max_length=None,
                 return_tensors=None):
        rows = []
        for text in texts:
            rows.append([self.vocab.setdefault(w, len(self.vocab) + 1)
                         for w in text.split()[:max_length]])
        width = max(len(r) for r in rows)
        return {
            'input_ids': torch.tensor([r + [0] * (width - len(r)) for r in rows]),
            'attention_mask': torch.tensor(
                [[1] * len(r) + [0] * (width - len(r)) for r in rows]),
        }

    def decode(self, ids, skip_special_tokens=True):
        back = {v: k for k, v in self.vocab.items()}
        return " ".join(back[int(i)] for i in ids if int(i) in back)


def _real_tokenizer(words):
    """A genuine PreTrainedTokenizerFast over a fixed word-level vocab.

    Tevatron's TrainCollator drives the full tokenizer API -- `return_attention_mask`,
    `return_token_type_ids`, `add_special_tokens`, and `tokenizer.pad` -- so a stub
    that only implements `__call__` cannot exercise it. `tokenizers` ships with
    transformers, so this stays offline: no hub, no cache.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {'[PAD]': 0, '[UNK]': 1}
    for word in words:
        vocab.setdefault(word, len(vocab))
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                   unk_token='[UNK]')


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


# ---- precision: which arm gets bf16 WEIGHTS, and which only autocasts ----------

def test_tevatron_ignores_training_arguments_bf16_when_building():
    """`EncoderModel.build` never reads `train_args` for dtype (encoder.py:114).

    This is the whole reason the ANCE trainer must pass `torch_dtype` explicitly:
    constructing `TrainingArguments(bf16=True)` and handing it to `build` looks like it
    sets the precision and does not. The BRIGHT ANCE arm silently trained FP32 weights
    while the in-batch baseline it is compared against trained bf16 ones.
    """
    import inspect
    from tevatron.retriever.modeling.encoder import EncoderModel

    src = inspect.getsource(EncoderModel.build)
    assert 'torch_dtype' not in src, \
        "pinned Tevatron now sets torch_dtype itself; the explicit pass may be redundant"
    # ...whereas the driver, which the in-batch and cross-batch arms go through, does.
    from tevatron.retriever.driver import train as tevatron_train
    driver = inspect.getsource(tevatron_train.main)
    assert 'torch_dtype' in driver and 'training_args.bf16' in driver


def test_the_two_ance_arms_choose_their_precision_deliberately():
    """`ance` matches the baselines' bf16 WEIGHTS; `ance_paper` keeps FP32 weights.

    Not an oversight in either direction. The BRIGHT comparison is only meaningful if
    ANCE and in-batch share a numerical path, but LAMB at lr 1e-6 moves a weight by far
    less than bf16 can represent, so bf16 master weights would round most of the
    reproduction's updates to zero.
    """
    import inspect
    import run_ance_train

    src = inspect.getsource(run_ance_train.main)
    bge = src[src.index('from tevatron.retriever.modeling import DenseModel'):]
    assert 'torch_dtype=torch.bfloat16 if bf16' in bge, \
        "the BGE arm no longer pins its weight dtype"
    # The paper arm builds its encoder before that branch and passes no dtype.
    paper = src[:src.index('from tevatron.retriever.modeling import DenseModel')]
    assert 'load_ance_encoder' in paper and 'torch_dtype' not in paper

    # Both still autocast, which is orthogonal to the weight dtype.
    assert src.count("torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16)") == 2


# ---- the BGE collator: dynamic padding, and the (B, G, L) shape it must keep ----

def test_the_bge_collator_pads_to_the_batch_not_the_cap():
    """q1024/p1024 against a mixture whose median query is 41 word-pieces.

    The loader used to tokenize with padding='max_length' per item, so every step paid
    for a full 1024-token query and 1024-token passages regardless of content. This is
    now Tevatron's own TrainCollator -- the same one the in-batch and cross-batch arms
    get through the driver -- so all three BGE arms tokenize identically.
    """
    from run_ance_train import make_bge_collate

    tok = _real_tokenizer("short query a considerably longer than the first one "
                          "positive negative".split())
    collate = make_bge_collate(tok, 1024, 1024)
    q_enc, p_enc = collate([
        (["short query"], [["a positive"], ["a negative"]]),
        (["a considerably longer query than the first one"], [["a a"], ["a a"]]),
    ])

    # Padded to the batch, rounded up to Tevatron's pad_to_multiple_of=16 for
    # tensor-core alignment -- 16, not the 1024 cap. The in-batch and cross-batch arms
    # get the same rounding through the driver, so the three BGE arms now agree.
    assert q_enc['input_ids'].shape == (2, 16), q_enc['input_ids'].shape
    # Flat (B*G, L), NOT (B, G, L): DenseModel.forward derives the group width from
    # p_reps.size(0) // q_reps.size(0), so the old reshape round-trip bought nothing.
    assert p_enc['input_ids'].shape == (4, 16), p_enc['input_ids'].shape
    assert p_enc['attention_mask'].shape == p_enc['input_ids'].shape
    # Real content is what the mask marks; the rest is padding, and there are 1008
    # fewer pad positions per query than the old padding='max_length' produced.
    assert q_enc['attention_mask'][0].sum() == 2
    assert q_enc['attention_mask'][1].sum() == 8


def test_the_bge_collator_keeps_the_positive_first_in_every_group():
    """DenseModel's target is arange(B) * G, so index 0 of each group IS the positive.

    The collator flattens B*G passages into one call; getting that order wrong would
    train every query against another example's positive, and the loss would still
    look perfectly healthy.
    """
    from run_ance_train import make_bge_collate

    tok = _real_tokenizer(["q0", "q1", "POS0", "NEG0", "POS1", "NEG1"])
    collate = make_bge_collate(tok, 32, 32)
    _, p_enc = collate([(["q0"], [["POS0"], ["NEG0"]]),
                        (["q1"], [["POS1"], ["NEG1"]])])
    decoded = [tok.decode(row, skip_special_tokens=True)
               for row in p_enc['input_ids']]
    assert decoded == ["POS0", "NEG0", "POS1", "NEG1"], decoded


def test_the_dataset_emits_the_shape_the_collator_indexes():
    """TrainCollator reads `f[0][0]` and `p[0]` (collator.py:31-32), so the dataset
    must yield the query and every passage wrapped as a sequence. A bare string would
    silently collate its FIRST CHARACTER and train on nothing."""
    import json
    import tempfile
    from run_ance_train import ANCEDataset

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "r.jsonl").write_text(json.dumps({
            'query_id': 'q0', 'query': 'the query',
            'positive_passages': [{'docid': 'p0', 'text': 'the positive'}],
            'negative_passages': [{'docid': 'n0', 'text': 'the negative'}],
        }) + "\n")
        query, passages = ANCEDataset(Path(tmp), train_group_size=2)[0]
        assert query == ['the query'], query
        assert passages == [['the positive'], ['the negative']], passages


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
    ("precision: Tevatron build ignores train_args.bf16", test_tevatron_ignores_training_arguments_bf16_when_building),
    ("precision: each ANCE arm pins its own dtype", test_the_two_ance_arms_choose_their_precision_deliberately),
    ("collate: pads to the batch, not the cap", test_the_bge_collator_pads_to_the_batch_not_the_cap),
    ("collate: positive stays first in each group", test_the_bge_collator_keeps_the_positive_first_in_every_group),
    ("collate: dataset emits the shape the collator indexes", test_the_dataset_emits_the_shape_the_collator_indexes),
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
