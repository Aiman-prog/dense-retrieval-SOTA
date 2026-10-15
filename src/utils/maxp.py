"""MaxP: a document is its first `model.max_passages` passages and scores as its best one.

ANCE's NLL_MultiChunk (model/models.py:84) takes the max over passage scores before
the loss, and its ANN export scores a document by its best chunk. Every MaxP consumer --
training (`scripts/maxp_training.py`), ANCE and GRASS mining, and evaluation -- splits,
encodes and ranks through this module, so a model is mined and evaluated on exactly the
passages it trained on.

Every artifact built from these representations records `representation_contract`;
a consumer that finds a different contract refuses the artifact rather than mixing a
FirstP cache into a MaxP run.
"""
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

SCHEMA = "maxp-v1"


class ContractError(RuntimeError):
    """An artifact was built under a different representation than this run uses."""


def tokenizer_digest(model_dir):
    """sha256 of what the tokenizer DOES: its vocabulary and special tokens.

    Not its files: save_pretrained re-serializes tokenizer_config.json (adding
    added_tokens_decoder), so every checkpoint's bytes differ from the base model's
    while tokenizing identically. Works for a directory or an offline repo id.
    """
    import json
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    payload = json.dumps({'vocab': tok.get_vocab(), 'special': tok.special_tokens_map},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def representation_contract(config, tokenizer_dir, data_sha256=None):
    """Everything that decides which vectors a document becomes."""
    m = config['model']
    if m.get('pooling', 'cls') != 'cls' or not m.get('normalize', False):
        raise ContractError("MaxP encodes normalized CLS; config.model says "
                            f"pooling={m.get('pooling')!r} normalize={m.get('normalize')!r}")
    return {'schema': SCHEMA, 'tokenizer_sha256': tokenizer_digest(tokenizer_dir),
            'query_max_len': m['query_max_len'], 'passage_max_len': m['passage_max_len'],
            'max_passages': m['max_passages'], 'overlap': 0, 'pooling': 'cls',
            'normalize': True, 'aggregation': 'max', 'data_sha256': data_sha256}


def check_contract(recorded, expected, what):
    """Raise unless `recorded` (from an artifact) equals this run's contract."""
    if recorded is None:
        raise ContractError(f"{what} carries no representation contract; it predates "
                            f"MaxP ({SCHEMA}) and cannot be reused")
    diff = sorted(k for k in set(recorded) | set(expected)
                  if recorded.get(k) != expected.get(k))
    if diff:
        raise ContractError(f"{what} was built under a different representation: "
                            + ", ".join(f"{k} {recorded.get(k)!r} != {expected.get(k)!r}"
                                        for k in diff))


def doc_text(doc):
    """A corpus or mixture document as Tevatron formats it, title first when present."""
    return doc['title'] + ' ' + doc['text'] if 'title' in doc else doc['text']


def split_passages(tokenizer, texts, max_len, max_passages):
    """`texts` -> per text, up to `max_passages` int32 id arrays of at most `max_len`.

    Tokenized ONCE and sliced on tokens, never on characters, and each slice gets its
    own special tokens: passage 0 is exactly the FirstP truncation of the document.
    """
    size = max_len - tokenizer.num_special_tokens_to_add()
    out = []
    for text, ids in zip(texts, tokenizer(list(texts), add_special_tokens=False,
                                          verbose=False)['input_ids']):
        if not ids:
            raise ValueError(f"document tokenizes to nothing, cannot encode: {text[:80]!r}")
        out.append([np.asarray(tokenizer.build_inputs_with_special_tokens(ids[i:i + size]),
                               dtype=np.int32)
                    for i in range(0, min(len(ids), size * max_passages), size)])
    return out


@torch.no_grad()
def encode_ids(encoder, tokenizer, id_arrays, batch_size, device):
    """Input-id arrays -> [n, D] float32 CPU tensor of normalized CLS, in input order.

    `encoder` is the bare HF model (`DenseModel.encoder`, or GRASS's AutoModel). Batches
    are length-sorted, so padding is short; it is never attended to either way.
    """
    order = np.argsort([len(a) for a in id_arrays], kind='stable')
    out = None
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tokenizer.pad({'input_ids': [id_arrays[i].tolist() for i in idx]},
                              return_tensors='pt')
        hidden = encoder(**{k: v.to(device) for k, v in batch.items()}).last_hidden_state
        reps = F.normalize(hidden[:, 0].float(), dim=-1).cpu()
        if out is None:
            out = torch.empty(len(id_arrays), reps.size(-1))
        out[torch.as_tensor(idx)] = reps
    return out


def encode_docs_maxp(encoder, tokenizer, texts, *, max_len, max_passages, batch_size,
                     device, T=None, dtype=None):
    """Documents -> passage vectors `[n, P, D]` (or `[T, n, P, D]` over `T` dropout
    passes) and the valid-passage mask `[n, P]`. Padded slots are zero and masked.

    With `T`, each pass is a genuine stochastic forward (`helpers.dropout_only`), so a
    document's `T` states are independent dropout samples, as `helpers.encode_mc`.
    """
    from utils.helpers import dropout_only
    split = split_passages(tokenizer, texts, max_len, max_passages)
    flat = [p for ps in split for p in ps]
    rows = torch.tensor([i for i, ps in enumerate(split) for _ in ps], dtype=torch.long)
    cols = torch.tensor([j for ps in split for j in range(len(ps))], dtype=torch.long)
    mask = torch.zeros(len(texts), max_passages, dtype=torch.bool)
    mask[rows, cols] = True

    def one_pass():
        vecs = encode_ids(encoder, tokenizer, flat, batch_size, device)
        z = torch.zeros(len(texts), max_passages, vecs.size(-1))
        z[rows, cols] = vecs
        return z

    if T is None:
        Z = one_pass()
    else:
        with dropout_only(encoder):
            Z = torch.stack([one_pass() for _ in range(int(T))])
    Z = Z.to(device=device, dtype=dtype or Z.dtype)
    n_pass = int(T or 1)
    stats = {'mc_passes': n_pass, 'passages': len(flat),
             'examples_encoded': len(flat) * n_pass,
             'forward_batches': -(-len(flat) // max(batch_size, 1)) * n_pass}
    return Z, mask.to(device), stats


def maxp_loss(q_reps, p_reps, *, doc_index, target, mask, temperature):
    """Contrastive loss over documents, each scoring as its best passage (`doc_index`
    maps passage -> document). Per query-document pair, so two queries can be won by
    different passages of one document; the gradient reaches only the winner.
    `mask[i, j]` drops document j from query i's denominator: another occurrence of
    its positive, or another known positive."""
    scores = q_reps.float() @ p_reps.float().T
    docs = scores.new_full(mask.shape, float('-inf')).scatter_reduce(
        1, doc_index.unsqueeze(0).expand_as(scores), scores, 'amax')
    return F.cross_entropy(docs.masked_fill(mask, float('-inf')) / temperature, target)


def collapse_to_docs(rows, scores, parents, depth):
    """Best-first passage hits -> ({docid: best passage score} for the top `depth`
    documents, complete).

    A document's first hit is its best passage. Once `depth` documents are seen, the
    cutoff score is known; hits tying it are still read, so a tie at the boundary is
    settled by docid (lower wins) rather than by FAISS's arbitrary order. `complete` is
    False when the hits ran out before a score below the cutoff -- a deeper search
    could still hold a qualifying document.
    """
    best, cutoff = {}, None
    for row, score in zip(rows, scores):
        score = float(score)
        if row < 0 or (cutoff is not None and score < cutoff):
            return _top(best, depth), True      # -1: FAISS ran past the index
        best.setdefault(parents[row], score)
        if cutoff is None and len(best) == depth:
            cutoff = score
    return _top(best, depth), False


def _top(best, depth):
    return dict(sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:depth])


def search_docs(index, query_embs, parents, depths, max_passages):
    """Document-level MaxP retrieval for every query: `depths[i]` documents each.

    Starts `max_passages` x the deepest request deep -- enough for that many documents,
    since each holds at most `max_passages` vectors -- and re-searches, deeper, only
    the queries whose boundary could still hide a tied document.
    """
    n = index.ntotal
    k = min(max_passages * max(depths), n)
    out, pending = [None] * len(depths), list(range(len(depths)))
    while pending:
        scores, rows = index.search(np.ascontiguousarray(query_embs[pending]), k)
        retry = []
        for local, qi in enumerate(pending):
            docs, complete = collapse_to_docs(rows[local], scores[local], parents,
                                              depths[qi])
            if complete or k >= n:
                out[qi] = docs
            else:
                retry.append(qi)
        pending, k = retry, min(2 * k, n)
    return out


def center(embs, centroid):
    """Subtract the corpus centroid and re-normalize; refuse a degenerate result."""
    x = np.asarray(embs, dtype=np.float32) - centroid
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or (norms == 0).any():
        raise ValueError("centering produced a non-finite or zero-norm vector")
    return (x / norms).astype(np.float32)
