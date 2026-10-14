"""MaxP: a document is its first `model.max_passages` passages and scores as its best one.

ANCE's NLL_MultiChunk (model/models.py:84) takes the max over passage scores before
the loss, and its ANN export scores a document by its best chunk. Training
(`scripts/train_maxp.py`) and evaluation (`src/evaluation/evaluate.py`) both split
through `split_passages`, so a model is evaluated on exactly the passages it trained on.
"""
import numpy as np
import torch
import torch.nn.functional as F


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


def maxp_doc_scores(q_reps, p_reps, doc_index, n_docs):
    """[B, N] query-passage scores -> [B, n_docs], each document its best passage.

    Per query-document pair, so two queries can be won by different passages of the
    same document. The gradient reaches only each pair's winning passage.
    """
    scores = q_reps.float() @ p_reps.float().T
    out = scores.new_full((scores.size(0), n_docs), float('-inf'))
    return out.scatter_reduce(1, doc_index.unsqueeze(0).expand_as(scores), scores, 'amax')


def maxp_loss(q_reps, p_reps, *, doc_index, target, mask, temperature):
    """Contrastive loss over documents. `mask[i, j]` drops document j from query i's
    denominator: another occurrence of its positive, or another known positive."""
    scores = maxp_doc_scores(q_reps, p_reps, doc_index, mask.size(1))
    return F.cross_entropy(scores.masked_fill(mask, float('-inf')) / temperature, target)


def collapse_to_docs(rows, scores, parents, depth):
    """Best-first passage hits -> {docid: best passage score}, for the top `depth` docs.

    A document's first hit is its best passage, so the first `depth` distinct parents
    are the document-level top `depth`. With at most P passages per document, a
    passage search of depth P * `depth` always reaches them. Ties go to the lower docid.
    """
    best = {}
    for row, score in zip(rows, scores):
        if row < 0:          # FAISS pads past the end of the index with -1
            break
        best.setdefault(parents[row], float(score))
        if len(best) == depth:
            break
    return dict(sorted(best.items(), key=lambda kv: (-kv[1], kv[0])))


@torch.no_grad()
def encode_ids(model, tokenizer, id_arrays, batch_size, device):
    """Input-id arrays -> [n, d] float32, in input order. Batches are length-sorted, so
    padding is short; it is never attended to either way."""
    order = np.argsort([len(a) for a in id_arrays], kind='stable')
    out = None
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tokenizer.pad({'input_ids': [id_arrays[i].tolist() for i in idx]},
                              return_tensors='pt')
        reps = model.encode_passage({k: v.to(device) for k, v in batch.items()})
        if out is None:
            out = np.empty((len(id_arrays), reps.size(-1)), dtype=np.float32)
        out[idx] = reps.float().cpu().numpy()
    return out
