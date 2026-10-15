"""The MaxP training step every arm shares: in-batch, cross-batch, BGE-ANCE, async GRASS.

A batch is B queries and their G documents each (positive first). Every document is
split by `utils.maxp.split_passages`, every passage encoded as normalized CLS, and each
document scores, per query, as its best passage before the temperature-scaled
contrastive loss. Known positives other than a query's target leave its denominator.

GradCache bounds memory: a batch of B*G documents carries between B*G and P*B*G
passages. Its RandContext replays the dropout RNG between the no-grad and the grad
pass. Cross-batch runs the same loss over the documents of every rank
(`distributed_maxp_loss`).

Lives in scripts/ because it is training machinery; `src/` must not import scripts/.
"""
import hashlib
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils.maxp import doc_text, split_passages, maxp_loss

CHUNK_ROW_SUFFIX = r'__c\d+$'     # what the rolled-back chunk split appended to ids


class NonFiniteOptimization(RuntimeError):
    """Optimization diverged; stopping rather than saving poisoned weights."""


class ClsEncoder(torch.nn.Module):
    """Normalized CLS of a bare HF encoder: the one representation every arm trains.

    Wrapping `DenseModel.encoder` and GRASS's AutoModel alike is what makes the four
    arms' passage vectors the same function of the weights.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # The CLS representation never reads the pooler, so its weights get no
        # gradient. Frozen, so DDP does not wait for gradients that never come
        # (Tevatron's Trainer used find_unused_parameters=True for the same reason).
        if getattr(encoder, 'pooler', None) is not None:
            encoder.pooler.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        return F.normalize(hidden[:, 0], dim=-1)


CONTRACT_NAME = "maxp_contract.json"


def write_contract(directory, contract):
    """Record the representation a checkpoint was trained under, beside its weights."""
    import json
    from utils.helpers import atomic_write
    with atomic_write(Path(directory) / CONTRACT_NAME) as f:
        json.dump(contract, f, indent=2)


def digest(text):
    """The text digest `preprocessor._derive` canonicalizes duplicate passages by."""
    return hashlib.md5(text.strip().encode()).hexdigest()


def _hash64(hex_digest):
    return int(hex_digest[:15], 16)          # < 2**60: fits int64 for all_gather


def known_positives(records, qrels_dict=None, corpus_lookup=None):
    """query text -> digests of every known positive.

    Mixture labels united with the canonical training qrels, and united across
    records with identical query text. Keyed by text, not id: the mixture holds raw
    docids that the derived qrels remap, and one question can appear under two ids.
    """
    known = {}
    for rec in records:
        found = known.setdefault(rec['query'], set())
        found.update(digest(doc_text(p)) for p in rec['positive_passages'])
        for docid in (qrels_dict or {}).get(str(rec.get('query_id')), ()):
            if corpus_lookup is not None and docid in corpus_lookup:
                found.add(digest(corpus_lookup[docid]))
    return known


def _plain(item):
    """(query, [docs]) from any row shape the arms produce: Tevatron's
    ((query, img, vid, aud), [(doc, ...)]), ANCE's ([query], [[doc]]), or plain text."""
    query, docs = item
    pick = lambda x: x if isinstance(x, str) else x[0]
    return pick(query), [pick(d) for d in docs]


def make_collate(tokenizer, max_q, max_p, max_passages, known):
    """A batch -> queries, flat passages sorted by length, and the document geometry.

    Row i's positive is document i*G. `mask[i, j]` is True when document j is a known
    positive of query i other than its target -- a duplicate of its own positive, or
    another positive of the same query -- so it leaves the denominator.
    """
    def collate(batch):
        rows = [_plain(item) for item in batch]
        queries = [q for q, _ in rows]
        docs = [d for _, ds in rows for d in ds]
        group = len(rows[0][1])
        split = split_passages(tokenizer, docs, max_p, max_passages)
        flat = sorted(((j, p) for j, ps in enumerate(split) for p in ps),
                      key=lambda t: len(t[1]))   # short chunks pad short (_split_trimmed)
        passage = tokenizer.pad({'input_ids': [p.tolist() for _, p in flat]},
                                return_tensors='pt')
        query = tokenizer(queries, truncation=True, max_length=max_q,
                          padding=True, return_tensors='pt')
        digests = [digest(d) for d in docs]
        target = [i * group for i in range(len(queries))]
        known_rows = [known.get(q, set()) for q in queries]
        mask = torch.tensor([[j != target[i] and digests[j] in known_rows[i]
                              for j in range(len(docs))] for i in range(len(queries))])
        return {'query': dict(query), 'passage': dict(passage),
                'doc_index': torch.tensor([j for j, _ in flat]),
                'target': torch.tensor(target), 'mask': mask,
                # cross-batch builds its mask across ranks from these instead
                'doc_hash': torch.tensor([_hash64(h) for h in digests]),
                'known_hash': [{_hash64(h) for h in k} for k in known_rows]}
    return collate


def _split_trimmed(model_input, chunk_size):
    """Split a flat input dict along the batch, each chunk cut to its longest row.

    Right padding, so trimming drops only pad columns. Passages arrive length-sorted,
    so a chunk of short passages no longer pays for the batch's longest one.
    """
    n = model_input['input_ids'].size(0)
    chunks = []
    for start in range(0, n, chunk_size):
        piece = {k: v[start:start + chunk_size] for k, v in model_input.items()}
        width = int(piece['attention_mask'].sum(1).max())
        chunks.append({k: v[:, :width] for k, v in piece.items()})
    return chunks


def _gather(t):
    """all_gather with this rank's slice kept LIVE, so its gradient survives."""
    parts = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, t.contiguous())
    parts[dist.get_rank()] = t
    return torch.cat(parts)


def distributed_maxp_loss(q_reps, p_reps, *, doc_index, target, doc_hash, known_hash,
                          temperature):
    """`maxp_loss` over the queries and documents of EVERY rank.

    Each rank holds B queries and B*G documents (equal across ranks: drop_last) but a
    different number of passages, so passages are padded to the largest rank's count
    for transport and padded rows are pointed at an extra document column that is
    dropped before the loss -- a pad can never be a candidate. Every rank computes the
    full loss, scaled by the world size, so DDP's gradient average restores it.
    """
    world, rank = dist.get_world_size(), dist.get_rank()
    B, D = q_reps.size(0), doc_hash.size(0)
    n = torch.tensor([p_reps.size(0)], device=p_reps.device)
    sizes = [torch.empty_like(n) for _ in range(world)]
    dist.all_gather(sizes, n)
    pad = int(max(s.item() for s in sizes)) - p_reps.size(0)
    pad_doc = world * D
    p_pad = torch.cat([p_reps, p_reps.new_zeros(pad, p_reps.size(1))])
    idx_pad = torch.cat([doc_index + rank * D,
                         doc_index.new_full((pad,), pad_doc)])
    all_q, all_p = _gather(q_reps), _gather(p_pad)
    all_idx, all_hash = _gather(idx_pad), _gather(doc_hash)
    all_target = _gather(target + rank * D)

    hashes = all_hash.tolist()
    local = torch.tensor([[j != int(all_target[rank * B + i]) and hashes[j] in known_hash[i]
                           for j in range(world * D)] for i in range(B)],
                         device=q_reps.device)
    mask = torch.cat([local, torch.ones(B, 1, dtype=torch.bool, device=local.device)],
                     dim=1)                     # the pad column is never a candidate
    all_mask = _gather(mask.to(torch.uint8)).bool()     # NCCL has no bool type
    loss = maxp_loss(all_q, all_p, doc_index=all_idx, target=all_target,
                     mask=all_mask, temperature=temperature)
    return loss * world


def make_grad_cache(model, q_chunk, p_chunk, temperature, distributed=False):
    """GradCache over `model` (a ClsEncoder, or its DDP wrapper) with the MaxP loss."""
    from grad_cache import GradCache
    if distributed:
        loss_fn = lambda q, p, **kw: distributed_maxp_loss(
            q, p, temperature=temperature, **kw)
    else:
        loss_fn = lambda q, p, doc_index, target, mask, **_: maxp_loss(
            q, p, doc_index=doc_index, target=target, mask=mask, temperature=temperature)
    return GradCache(models=[model, model], chunk_sizes=[q_chunk, p_chunk],
                     loss_fn=loss_fn, split_input_fn=_split_trimmed, fp16=False)


def train_step(gc, model, optimizer, scheduler, batch, max_grad_norm, step,
               distributed=False):
    """One GradCache step; finite checks before the optimizer moves anything."""
    device = next(model.parameters()).device
    move = lambda v: ({k: t.to(device) for k, t in v.items()} if isinstance(v, dict)
                      else v.to(device) if torch.is_tensor(v) else v)
    b = {k: move(v) for k, v in batch.items()}
    kw = ({'doc_hash': b['doc_hash'], 'known_hash': b['known_hash']} if distributed
          else {'mask': b['mask']})
    loss = float(gc(b['query'], b['passage'], no_sync_except_last=distributed,
                    doc_index=b['doc_index'], target=b['target'], **kw))
    if not math.isfinite(loss):
        raise NonFiniteOptimization(f"non-finite loss={loss} at step {step}; no step taken.")
    try:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                                   error_if_nonfinite=True)
    except RuntimeError as exc:
        raise NonFiniteOptimization(f"non-finite gradient norm at step {step}: {exc}") from exc
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return loss, float(grad_norm)
