"""
MaxP tests (src/utils/maxp.py, scripts/train_maxp.py): one per requirement.

  1. the splitter: own special tokens per passage, the cap, passage 0 == FirstP
  2. the loss: per-query max over passages, known positives masked
  3. GradCache: same gradients as the uncached loss, dropout replayed
  4. retrieval: passage search + collapse == brute-force document MaxP after exclusions

CPU only. Test 1 needs the BGE-M3 tokenizer in the local HF cache.
Run: python tests/maxp_test.py      (KMP_DUPLICATE_LIB_OK=TRUE on macOS)
"""
import os
import sys
import traceback
from pathlib import Path

# macOS ships faiss and torch with two OpenMP runtimes: torch-before-faiss segfaults a
# faiss search, and multithreaded torch next to faiss deadlocks or segfaults.
os.environ.setdefault('OMP_NUM_THREADS', '1')
import faiss  # noqa: F401,E402
import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.maxp import split_passages, maxp_loss, collapse_to_docs  # noqa: E402
from utils.helpers import apply_exclusions, search_depth  # noqa: E402


def test_splitter():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('BAAI/bge-m3', local_files_only=True)
    text = ' '.join(f'word{i}' for i in range(3000))          # far beyond 4 x 16 tokens
    passages, = split_passages(tok, [text], 16, 4)
    assert len(passages) == 4, len(passages)
    for p in passages:
        assert len(p) <= 16 and p[0] == tok.cls_token_id and p[-1] == tok.sep_token_id, p
    firstp = tok(text, truncation=True, max_length=16)['input_ids']
    assert passages[0].tolist() == firstp, (passages[0].tolist(), firstp)
    short, = split_passages(tok, ['a short document'], 16, 4)
    assert len(short) == 1


def test_loss():
    q = torch.tensor([[1., 0.], [0., 1.]])
    p = torch.tensor([[1., 0.], [0., 1.], [.5, .5], [0., -1.], [.9, .1]], requires_grad=True)
    doc_index = torch.tensor([0, 0, 1, 2, 3])                 # doc 0 has two passages
    target = torch.tensor([0, 2])
    mask = torch.tensor([[False, False, False, True], [False] * 4])   # doc 3 known to q0
    loss = maxp_loss(q, p, doc_index=doc_index, target=target, mask=mask, temperature=0.5)
    # q0: doc0 = max(1, 0) = 1, doc1 = .5, doc2 = 0, doc3 masked
    # q1: doc0 = max(0, 1) = 1 (a DIFFERENT passage wins), doc1 = .5, doc2 = -1, doc3 = .1
    expected = (F.cross_entropy(torch.tensor([[1., .5, 0.]]) / .5, torch.tensor([0]))
                + F.cross_entropy(torch.tensor([[1., .5, -1., .1]]) / .5, torch.tensor([2]))) / 2
    assert torch.isclose(loss, expected), (loss, expected)


def test_grad_cache_matches_direct():
    from transformers import XLMRobertaConfig, XLMRobertaModel
    from tevatron.retriever.modeling import DenseModel
    from tevatron.retriever.gc_trainer import get_dense_rep
    from train_maxp import make_grad_cache, _split_trimmed

    torch.manual_seed(0)
    encoder = XLMRobertaModel(XLMRobertaConfig(
        vocab_size=50, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=40), add_pooling_layer=False)
    model = DenseModel(encoder, pooling='cls', normalize=True, temperature=0.02).train()

    def ids(lengths):
        x = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
        for i, n in enumerate(lengths):
            x[i, :n] = torch.randint(5, 50, (n,))
        return {'input_ids': x, 'attention_mask': (x > 0).long()}
    query, passage = ids([6, 9, 4]), ids([3, 5, 5, 8, 8, 12, 12])
    kw = dict(doc_index=torch.tensor([0, 1, 1, 2, 3, 4, 5]), target=torch.tensor([0, 2, 4]),
              mask=torch.zeros(3, 6, dtype=torch.bool))
    recipe = {'gc_q_chunk_size': 2, 'gc_p_chunk_size': 3}

    def grads():
        g = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        model.zero_grad(set_to_none=True)
        return g

    # Reference: the same chunks forwarded WITH grad, in GradCache's RNG order.
    torch.manual_seed(1)
    reps = [torch.cat([get_dense_rep(model(**c)) for c in _split_trimmed({k: v}, n)])
            for (k, v), n in ((('query', query), 2), (('passage', passage), 3))]
    direct = maxp_loss(*reps, temperature=0.02, **kw)
    direct.backward()
    want = grads()

    torch.manual_seed(1)
    cached = make_grad_cache(model, recipe, 0.02)({'query': query}, {'passage': passage}, **kw)
    got = grads()
    assert torch.isclose(cached, direct.detach(), atol=1e-6), (cached, direct)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(want, got))


def test_retrieval_matches_brute_force():
    rng = np.random.default_rng(0)
    n_docs, max_p, k = 60, 4, 10
    counts = rng.integers(1, max_p + 1, n_docs)
    parents = [f'd{j:02d}' for j, c in enumerate(counts) for _ in range(c)]
    embs = rng.normal(size=(len(parents), 8)).astype(np.float32)
    queries = rng.normal(size=(5, 8)).astype(np.float32)
    qids = [f'q{i}' for i in range(5)]
    excluded = {q: frozenset(rng.choice([f'd{j:02d}' for j in range(n_docs)], 7,
                                        replace=False)) for q in qids}

    index = faiss.IndexFlatIP(8)
    index.add(embs)
    depth = search_depth(k, excluded)
    scores, rows = index.search(queries, max_p * depth)
    run = {q: collapse_to_docs(r, s, parents, search_depth(k, excluded, q))
           for q, r, s in zip(qids, rows, scores)}
    got = apply_exclusions(run, excluded, k)

    for q, vec in zip(qids, queries):
        best = {}
        for parent, e in zip(parents, embs):
            best[parent] = max(best.get(parent, -np.inf), float(vec @ e))
        want = sorted((d for d in best if d not in excluded[q]), key=lambda d: -best[d])[:k]
        assert list(got[q]) == want, (q, list(got[q]), want)
        assert all(np.isclose(got[q][d], best[d], atol=1e-5) for d in want)


TESTS = [
    ("splitter: own special tokens, cap, passage 0 == FirstP", test_splitter),
    ("loss: per-query max, known positives masked", test_loss),
    ("GradCache: gradients match the direct loss", test_grad_cache_matches_direct),
    ("retrieval: collapse == brute-force document MaxP", test_retrieval_matches_brute_force),
]


def _run(name, fn):
    try:
        fn()
    except Exception as e:                                    # noqa: BLE001
        print(f"  ❌ {name}\n       {type(e).__name__}: {e}")
        if os.environ.get("MAXP_TEST_TRACE"):
            traceback.print_exc()
        return False
    print(f"  ✅ {name}")
    return True


def main():
    print("\nMaxP tests")
    print("=" * 62)
    passed = sum(_run(name, fn) for name, fn in TESTS)
    print("=" * 62)
    print(f"  {passed}/{len(TESTS)} passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
