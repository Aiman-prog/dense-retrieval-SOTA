"""
MaxP tests: one per requirement.

  1. splitter: own special tokens, cap, passage 0 == FirstP; encoded mask/padding
  2. loss: per-query max over passages, known positives masked
  3. GradCache: same gradients as the uncached loss, dropout replayed
  4. cross-batch: 2-rank distributed gradients == one logical batch
  5. retrieval: depth, exclusions, boundary ties, exhaustion; ANCE mines documents
  6. GRASS MC scoring: per-pass max, then moments (winner changes across passes)
  7. GRASS cache: replacement rewrites a whole slot; save/load; legacy rejected
  8. guards: contract mismatch, missing data manifest, chunk-row ids; centering

CPU only. Test 1 needs the BGE-M3 tokenizer in the local HF cache.
Run: python tests/maxp_test.py      (KMP_DUPLICATE_LIB_OK=TRUE on macOS)
"""
import os
import sys
import json
import random
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

# macOS ships faiss and torch with two OpenMP runtimes: torch-before-faiss segfaults a
# faiss search, and multithreaded torch next to faiss deadlocks or segfaults.
os.environ.setdefault('OMP_NUM_THREADS', '1')
import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.maxp import (split_passages, encode_docs_maxp, maxp_loss, search_docs,  # noqa: E402
                        center, check_contract, ContractError)
from utils.helpers import apply_exclusions, search_depth  # noqa: E402
from maxp_training import (ClsEncoder, make_grad_cache, _split_trimmed,  # noqa: E402
                           CHUNK_ROW_SUFFIX)


def _tiny_encoder(dropout=0.1, seed=0):
    from transformers import XLMRobertaConfig, XLMRobertaModel
    torch.manual_seed(seed)
    return ClsEncoder(XLMRobertaModel(XLMRobertaConfig(
        vocab_size=50, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=40, hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout)))     # with its pooler, as BGE-M3 loads


def _ids(lengths, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
    for i, n in enumerate(lengths):
        x[i, :n] = torch.randint(5, 50, (n,), generator=g)
    return {'input_ids': x, 'attention_mask': (x > 0).long()}


def _grads(module):
    g = [p.grad.clone() for p in module.parameters() if p.grad is not None]
    module.zero_grad(set_to_none=True)
    return g


def test_splitter():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('BAAI/bge-m3', local_files_only=True)
    text = ' '.join(f'word{i}' for i in range(3000))           # far beyond 4 x 16 tokens
    passages, short = split_passages(tok, [text, 'a short document'], 16, 4)
    assert len(passages) == 4 and len(short) == 1
    assert all(len(p) <= 16 and p[0] == tok.cls_token_id and p[-1] == tok.sep_token_id
               for p in passages)
    assert passages[0].tolist() == tok(text, truncation=True, max_length=16)['input_ids']
    # encoded: one row per document, valid passages marked, padding zero
    emb = torch.nn.Embedding(len(tok), 8)
    fake = lambda input_ids, attention_mask: SimpleNamespace(last_hidden_state=emb(input_ids))
    Z, mask, _ = encode_docs_maxp(fake, tok, [text, 'a short document'], max_len=16,
                                  max_passages=4, batch_size=3, device='cpu')
    assert Z.shape == (2, 4, 8) and mask.tolist() == [[True] * 4, [True, False, False, False]]
    assert Z[1, 1:].abs().sum() == 0
    want = F.normalize(emb(torch.as_tensor(passages[2]).long())[0], dim=-1)
    assert torch.allclose(Z[0, 2], want, atol=1e-6)


def test_loss():
    q = torch.tensor([[1., 0.], [0., 1.]])
    p = torch.tensor([[1., 0.], [0., 1.], [.5, .5], [0., -1.], [.9, .1]])
    doc_index = torch.tensor([0, 0, 1, 2, 3])                 # doc 0 has two passages
    mask = torch.tensor([[False, False, False, True], [False] * 4])   # doc 3 known to q0
    loss = maxp_loss(q, p, doc_index=doc_index, target=torch.tensor([0, 2]), mask=mask,
                     temperature=0.5)
    # q0: doc0 = max(1, 0) = 1, doc1 = .5, doc2 = 0, doc3 masked
    # q1: doc0 = max(0, 1) = 1 (a DIFFERENT passage wins), doc1 = .5, doc2 = -1, doc3 = .1
    expected = (F.cross_entropy(torch.tensor([[1., .5, 0.]]) / .5, torch.tensor([0]))
                + F.cross_entropy(torch.tensor([[1., .5, -1., .1]]) / .5, torch.tensor([2]))) / 2
    assert torch.isclose(loss, expected), (loss, expected)


def test_grad_cache_matches_direct():
    enc = _tiny_encoder().train()
    query, passage = _ids([6, 9, 4]), _ids([3, 5, 5, 8, 8, 12, 12], seed=1)
    kw = dict(doc_index=torch.tensor([0, 1, 1, 2, 3, 4, 5]), target=torch.tensor([0, 2, 4]),
              mask=torch.zeros(3, 6, dtype=torch.bool))
    # reference: the same chunks forwarded WITH grad, in GradCache's RNG order
    torch.manual_seed(1)
    reps = [torch.cat([enc(**c) for c in _split_trimmed(x, n)])
            for x, n in ((query, 2), (passage, 3))]
    direct = maxp_loss(*reps, temperature=0.02, **kw)
    direct.backward()
    want = _grads(enc)
    torch.manual_seed(1)
    cached = make_grad_cache(enc, 2, 3, 0.02)(query, passage, **kw)
    assert torch.isclose(cached, direct.detach(), atol=1e-6)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(want, _grads(enc)))


# Two ranks of 2 queries x 2 docs each. Rank 1's doc 2 has the text of rank 0 query
# 0's alternate known positive, so the cross-rank mask must drop it.
_Q = [[6, 9], [4, 7]]
_P = [[3, 5, 5, 8, 2], [8, 12, 12, 4, 6]]
_DI = [[0, 1, 1, 2, 3], [0, 0, 1, 2, 3]]
_HASH = [[11, 12, 13, 14], [21, 22, 99, 24]]
_KNOWN = [[{11, 99}, {13}], [{21}, {99}]]


def _dist_worker(rank, port, out):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank,
                            world_size=2)
    enc = _tiny_encoder(dropout=0.0)
    gc = make_grad_cache(DistributedDataParallel(enc), 1, 2, 0.05, distributed=True)
    gc(_ids(_Q[rank], seed=rank), _ids(_P[rank], seed=10 + rank), no_sync_except_last=True,
       doc_index=torch.tensor(_DI[rank]), target=torch.tensor([0, 2]),
       doc_hash=torch.tensor(_HASH[rank]), known_hash=_KNOWN[rank])
    if rank == 0:
        torch.save(_grads(enc), out)
    dist.destroy_process_group()


def test_crossbatch_distributed_matches_logical_batch():
    import socket
    import torch.multiprocessing as mp
    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo0' if sys.platform == 'darwin' else 'lo')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'grads.pt'
        mp.spawn(_dist_worker, args=(port, str(out)), nprocs=2)
        got = torch.load(out)
    # one logical batch: 4 queries x 8 docs, rank 1's docs offset by 4
    enc = _tiny_encoder(dropout=0.0)
    q = torch.cat([enc(**_ids(_Q[r], seed=r)) for r in (0, 1)])
    p = torch.cat([enc(**_ids(_P[r], seed=10 + r)) for r in (0, 1)])
    doc_index = torch.tensor(_DI[0] + [d + 4 for d in _DI[1]])
    target = torch.tensor([0, 2, 4, 6])
    hashes = _HASH[0] + _HASH[1]
    known = _KNOWN[0] + _KNOWN[1]
    mask = torch.tensor([[j != target[i] and hashes[j] in known[i] for j in range(8)]
                         for i in range(4)])
    assert mask[0, 6]                                    # the cross-rank positive
    maxp_loss(q, p, doc_index=doc_index, target=target, mask=mask,
              temperature=0.05).backward()
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(_grads(enc), got))


def test_retrieval_and_ance_mining():
    rng = np.random.default_rng(0)
    n_docs, P, k = 60, 4, 10
    counts = rng.integers(1, P + 1, n_docs)
    parents = [f'd{j:02d}' for j, c in enumerate(counts) for _ in range(c)]
    embs = rng.normal(size=(len(parents), 8)).astype(np.float32)
    queries = rng.normal(size=(5, 8)).astype(np.float32)
    qids = [f'q{i}' for i in range(5)]
    excluded = {q: frozenset(rng.choice([f'd{j:02d}' for j in range(n_docs)], 7,
                                        replace=False)) for q in qids}
    index = faiss.IndexFlatIP(8)
    index.add(embs)
    ranked = search_docs(index, queries, parents,
                         [search_depth(k, excluded, q) for q in qids], P)
    got = apply_exclusions(dict(zip(qids, ranked)), excluded, k)
    for q, vec in zip(qids, queries):
        best = {}
        for parent, e in zip(parents, embs):
            best[parent] = max(best.get(parent, -np.inf), float(vec @ e))
        want = sorted((d for d in best if d not in excluded[q]), key=lambda d: -best[d])[:k]
        assert list(got[q]) == want, (q, list(got[q]), want)

    # boundary tie: 'b' and 'a' tie at the cutoff, behind a shallow first search; the
    # deterministic policy (lower docid) must win wherever FAISS put them
    tie = np.array([[1, 0], [.5, .5], [.5, .5], [.5, .5], [0, 1]], dtype=np.float32)
    idx = faiss.IndexFlatIP(2)
    idx.add(tie)
    out = search_docs(idx, np.array([[1, 0]], dtype=np.float32),
                      ['x', 'y', 'b', 'a', 'z'], [2], 1)
    assert list(out[0]) == ['x', 'a'], out
    # exhaustion: asking for more documents than exist returns them all
    assert len(search_docs(idx, np.array([[1, 0]], dtype=np.float32),
                           ['x', 'y', 'b', 'a', 'z'], [9], 1)[0]) == 5

    # ANCE mining on a passage index: distinct parents, never a positive
    from ance_mining import mine_from_index
    with tempfile.TemporaryDirectory() as tmp:
        mix = Path(tmp) / 'm.jsonl'
        mix.write_text(''.join(json.dumps({'query_id': q, 'query': q, 'positive_passages':
                                           [{'docid': 'd00', 'text': 't'}]}) + '\n'
                               for q in qids))
        mined, failures, _, _ = mine_from_index(
            index, parents, (queries, qids), [mix], {}, n_negs=5, mining_depth=12,
            rng=random.Random(0), max_passages=P, search_batch_size=2)
    assert not failures and set(mined) == set(qids)
    assert all(len(set(v)) == 5 and 'd00' not in v for v in mined.values())


def test_mc_scoring_max_per_pass():
    from utils.negative_cache import score_cached_mcdp
    torch.manual_seed(0)
    T, B, P, D = 3, 4, 3, 5
    q = F.normalize(torch.randn(T, 2, D), dim=-1)
    Z = F.normalize(torch.randn(T, B, P, D), dim=-1)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    Z[:, ~mask] = 0
    g, s_hat, sigma = score_cached_mcdp(q, Z, 0.5, chunk_size=3, passage_mask=mask)
    per_pass = torch.tensor([[[max(float(q[t, i] @ Z[t, d, p]) for p in range(P)
                                   if mask[d, p]) for d in range(B)]
                              for i in range(2)] for t in range(T)])
    assert torch.allclose(s_hat, per_pass.mean(0), atol=1e-6)
    assert torch.allclose(sigma, per_pass.std(0, unbiased=False), atol=1e-6)
    assert torch.allclose(g, s_hat + 0.5 * sigma, atol=1e-6)
    winners = torch.stack([torch.stack([(q[t, 0] @ Z[t, 0, p]) for p in range(P)]).argmax()
                           for t in range(T)])
    assert len(set(winners.tolist())) > 1        # the winning passage moves across passes


def test_grass_cache_slots_and_persistence():
    from utils.negative_cache import NegativeCache
    cfg = {'B_doc': 3, 'cache_init_seed': 0, 'uncertainty': 'mcdp', 'max_passages': 2,
           'replacement_candidate_multiplier': 1, 'lambda_val': 0.0, 'reentry_top_k': 1}
    cache = NegativeCache.init_uniform(None, ['a', 'b', 'c', 'd', 'e'], cfg, torch.device('cpu'),
                                       dim=4)
    Z = torch.ones(2, 3, 2, 4)
    cache.attach_mc_states(Z, torch.ones(3, 2, dtype=torch.bool))
    # replacement: a one-passage candidate must clear the slot's second passage
    cand = [d for d in 'abcde' if d not in cache.docids]
    cache._sample_uniform = lambda *a, **k: cand[:1]
    cache._encode_mc_docs = lambda texts, *a: (
        torch.full((2, 1, 2, 4), 2.0) * torch.tensor([1., 0.]).view(1, 1, 2, 1),
        torch.tensor([[True, False]]), {'examples_encoded': 2, 'forward_batches': 2})
    out = cache._replace_cached_mcdp(torch.tensor([1]), None, None, {d: d for d in 'abcde'},
                                     list('abcde'), torch.ones(2, 1, 4), ['q'], 7, 2, cfg,
                                     torch.device('cpu'), {})
    assert out['num_replace'] == 1 and cache.docids[1] == cand[0]
    assert cache.passage_mask[1].tolist() == [True, False]
    assert cache.Z_mc[:, 1, 1].abs().sum() == 0 and torch.equal(cache.Z_student[1], cache.Z_mc[:, 1].mean(0))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'cache.pt'
        cache.save_state(path)
        back = NegativeCache.load_state(path, cfg, torch.device('cpu'))
        assert torch.equal(back.passage_mask, cache.passage_mask)
        for wrong in ({**cfg, 'max_passages': 1}, {**cfg, 'representation': {'schema': 'x'}}):
            try:
                NegativeCache.load_state(path, wrong, torch.device('cpu'))
            except (ValueError, ContractError):
                continue
            raise AssertionError(f"loaded under a different representation: {wrong}")


def test_guards_and_centering():
    from utils.helpers import require_training_data
    import re
    contract = {'schema': 'maxp-v1', 'max_passages': 4}
    for recorded in (None, {**contract, 'max_passages': 1}):
        try:
            check_contract(recorded, contract, 'artifact')
        except ContractError:
            continue
        raise AssertionError(f"accepted {recorded}")
    with tempfile.TemporaryDirectory() as tmp:
        for manifest in (None, {'chunk_rows': 3, 'files': {}}):
            if manifest is not None:
                (Path(tmp) / 'training_data_manifest.json').write_text(json.dumps(manifest))
            try:
                require_training_data(tmp)
            except (FileNotFoundError, ValueError):
                continue
            raise AssertionError(f"accepted training data with manifest {manifest}")
    # a re-saved tokenizer (every checkpoint) and the repo id are the same tokenizer
    from transformers import AutoTokenizer
    from utils.maxp import tokenizer_digest
    with tempfile.TemporaryDirectory() as tmp:
        AutoTokenizer.from_pretrained('BAAI/bge-m3', local_files_only=True).save_pretrained(tmp)
        assert tokenizer_digest(tmp) == tokenizer_digest('BAAI/bge-m3')
    assert re.search(CHUNK_ROW_SUFFIX, 'reasonir_hq_0__c3')
    assert not re.search(CHUNK_ROW_SUFFIX, 'reasonir_hq_0')
    x = np.array([[1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)
    c = center(x, x.mean(0))
    assert np.allclose(np.linalg.norm(c, axis=1), 1)
    try:
        center(x[:1], x[0])
    except ValueError:
        return
    raise AssertionError("a zero-norm centered vector was accepted")


TESTS = [
    ("splitter: special tokens, cap, FirstP, encoded mask", test_splitter),
    ("loss: per-query max, known positives masked", test_loss),
    ("GradCache: gradients match the direct loss", test_grad_cache_matches_direct),
    ("cross-batch: 2 ranks == one logical batch", test_crossbatch_distributed_matches_logical_batch),
    ("retrieval: depth, exclusions, ties, exhaustion; ANCE mining", test_retrieval_and_ance_mining),
    ("GRASS MC scoring: max per pass, then moments", test_mc_scoring_max_per_pass),
    ("GRASS cache: slot replacement, save/load, legacy refused", test_grass_cache_slots_and_persistence),
    ("guards: contract, data manifest, chunk ids; centering", test_guards_and_centering),
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
