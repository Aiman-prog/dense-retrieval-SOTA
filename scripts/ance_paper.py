"""Paper-fidelity ANCE: the four things the shared BGE-M3 machinery cannot express.

A *reproduction* of Microsoft's MS MARCO passage result, not a GRASS control arm. The
BRIGHT ANCE row deliberately keeps GRASS's objective so that table isolates negative
SELECTION; this module exists so the same mining pipeline can also run the paper's own
recipe, which is what makes "is this really ANCE?" answerable.

Everything here is pinned against the reference implementation, read from source:

    architecture   model/models.py:137-157  RobertaDot_NLL_LN
                   roberta -> Linear(hidden, 768) -> LayerNorm(768) on CLS
                   (masked_mean_or_first with use_mean=False is emb_all[0][:, 0])
    loss           model/models.py:77-81
                   cat([(q*pos).sum(-1), (q*neg).sum(-1)], 1) -> log_softmax -> -[:, 0]
                   raw dot product: NOT normalized, NO temperature
    optimizer      utils/lamb.py            LAMB, four quirks, see Lamb below
    mining         run_ann_data_gen.py:366-389, commands/run_train.sh:93
                   --topk_training 200 --negative_sample 20; shuffle the full top-200
                   and take the first 20 non-positive, non-duplicate candidates
    consumption    data/msmarco_data.py:337-362, run_train.sh:110 (--triplet)
                   one (query, positive, negative) per negative = 20 triplets per query

`select_ance_negatives` in `ance_mining.py` already implements the mining procedure
exactly; the paper recipe just sets n_negs=20 instead of 1. Nothing here duplicates it.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, RobertaConfig, RobertaModel

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

EMBED_DIM = 768
# transformers 2.3.0 wrote these checkpoints through RobertaForSequenceClassification,
# which persists a classification head ANCE never calls. Known-unused and known-present,
# so it is allowed through as an ALLOWLIST -- any other unexpected key is a hard error.
ALLOWED_UNEXPECTED_PREFIXES = ("classifier.",)


class AnceEncoder(PreTrainedModel):
    """RoBERTa + projection head, in the reference implementation's own key names.

    The attribute names are load-bearing, not cosmetic: `roberta`, `embeddingHead` and
    `norm` are exactly the keys in Microsoft's released state dict, so
    `from_pretrained` maps their checkpoint onto this class with no conversion step and
    `save_pretrained` writes all of it back into one weight file.

    Tevatron cannot host this. `DenseModel.build()` goes through `AutoModel`, which
    yields a bare RobertaModel and SILENTLY drops embeddingHead/norm -- the run would
    then train a random projection and converge to something that is not ANCE.
    """

    config_class = RobertaConfig
    base_model_prefix = 'ance_encoder'

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.embeddingHead = nn.Linear(config.hidden_size, EMBED_DIM)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.post_init()

    def _init_weights(self, module):
        """Upstream's head is normal_(0, initializer_range): RobertaDot_NLL_LN inherits
        RobertaPreTrainedModel, which supplies exactly this. Subclassing PreTrainedModel
        directly makes `post_init` a NO-OP, so without this `embeddingHead` would get
        nn.Linear's default kaiming-uniform instead. Only the warm-up ever observes a
        random head -- every other path loads one -- but the warm-up IS the artifact
        that defines it.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.roberta(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        return self.norm(self.embeddingHead(hidden[:, 0]))


def load_ance_encoder(model_path, **hf_kwargs):
    """Load a checkpoint and REFUSE anything left at its random initialization.

    `from_pretrained` only warns about missing and unexpected keys. A warning in a
    24-hour job's log is not a gate, and a silently reinitialized projection head is
    the most plausible route to a quietly wrong reproduction: the run trains, the loss
    falls, and the number simply is not ANCE.
    """
    model, info = AnceEncoder.from_pretrained(
        str(model_path), output_loading_info=True, **hf_kwargs)
    missing = [k for k in info.get('missing_keys', ()) if not k.endswith('position_ids')]
    unexpected = [k for k in info.get('unexpected_keys', ())
                  if not k.endswith('position_ids')
                  and not k.startswith(ALLOWED_UNEXPECTED_PREFIXES)]
    if missing or unexpected:
        raise ValueError(
            f"{model_path} does not match the ANCE architecture: missing "
            f"{missing[:10]}, unexpected {unexpected[:10]}. Missing keys would be left "
            f"randomly initialized; an unexpected key means this is not the checkpoint "
            f"it claims to be. Only {list(ALLOWED_UNEXPECTED_PREFIXES)} is allowed "
            f"through, because transformers 2.3.0 persisted an unused classification "
            f"head.")
    return model


# The projection head does not exist in roberta-base; the BM25 warm-up is where it is
# first trained. These four keys are therefore EXPECTED to be fresh there, and only there.
WARMUP_FRESH_KEYS = frozenset({'embeddingHead.weight', 'embeddingHead.bias',
                               'norm.weight', 'norm.bias'})
# roberta-base is published as a masked-LM checkpoint, so loading it as a RobertaModel
# leaves `lm_head.*` unconsumed and the pooler unfilled. Both are genuinely irrelevant
# here: AnceEncoder.forward reads last_hidden_state[:, 0] and never calls the pooler.
# Narrow allowlists, so any OTHER absent tensor is still a hard error.
WARMUP_ALLOWED_UNEXPECTED = ALLOWED_UNEXPECTED_PREFIXES + ("lm_head.",)
WARMUP_TOLERATED_MISSING = ("pooler.",)


def _tolerated_missing(key):
    return key.endswith('position_ids') or key.startswith(WARMUP_TOLERATED_MISSING)


def build_ance_encoder_from_base(base_model, **hf_kwargs):
    """roberta-base plus a FRESH projection head, for the BM25 warm-up only.

    The inverse of `load_ance_encoder`, and deliberately not a relaxation of it. That
    function refuses every missing key because an ANCE run must start from a trained
    head; the warm-up is what trains that head, so here it is expected to be new.

    The body is loaded EXPLICITLY rather than through `AnceEncoder.from_pretrained`,
    because that does not work and does not say so. `AnceEncoder` declares
    `base_model_prefix = 'ance_encoder'`, so HF cannot map a bare RobertaModel
    checkpoint's unprefixed keys (`embeddings.*`, `encoder.*`) onto `self.roberta.*`:
    it reports the ENTIRE body as newly initialized -- measured, all 28 tensors of a
    1-layer model -- and hands back a random encoder. That encoder trains, its loss
    falls, and it warms up nothing. Loading the body as a RobertaModel and copying it
    in is the only route that provably transfers the pretrained weights, and
    `load_state_dict` reports any key that did not.
    """
    from transformers import RobertaModel
    config = RobertaConfig.from_pretrained(str(base_model), **hf_kwargs)
    model = AnceEncoder(config)
    # output_loading_info, not just load_state_dict afterwards: from_pretrained fills a
    # key the checkpoint lacks with a random tensor, so by the time the state dict comes
    # back it is COMPLETE and a later strict check sees nothing wrong. This is the only
    # place the difference is still visible.
    body, info = RobertaModel.from_pretrained(str(base_model),
                                              output_loading_info=True, **hf_kwargs)
    missing = [k for k in info.get('missing_keys', ())
               if not _tolerated_missing(k)]
    unexpected = [k for k in info.get('unexpected_keys', ())
                  if not k.endswith('position_ids')
                  and not k.startswith(WARMUP_ALLOWED_UNEXPECTED)]
    if missing or unexpected:
        raise ValueError(
            f"{base_model} is not a usable warm-up base: missing {missing[:10]}, "
            f"unexpected {unexpected[:10]}. A missing body tensor stays randomly "
            f"initialized, which produces a healthy-looking loss curve and a warm-up "
            f"carrying no pretrained knowledge.")
    result = model.roberta.load_state_dict(body.state_dict(), strict=False)
    stranded = [k for k in result.missing_keys if not _tolerated_missing(k)]
    if stranded or result.unexpected_keys:
        raise ValueError(
            f"the RoBERTa body does not fit AnceEncoder.roberta: missing {stranded[:10]}, "
            f"unexpected {list(result.unexpected_keys)[:10]}.")
    return model


def pairwise_nll(q, pos, neg):
    """-log_softmax([q.pos, q.neg], dim=1)[:, 0], mean over the batch.

    Raw dot product. Normalizing would make it cosine and a temperature would rescale
    the logits; either is a different objective, so both are pinned by test.
    """
    logits = torch.cat([(q * pos).sum(-1, keepdim=True),
                        (q * neg).sum(-1, keepdim=True)], dim=1)
    return (-F.log_softmax(logits, dim=1)[:, 0]).mean()


class Lamb(torch.optim.Optimizer):
    """A direct port of the reference `utils/lamb.py`. Every quirk changes the update.

    * **no debiasing** -- upstream: "Paper v3 does not use debiasing", so step_size is
      the raw learning rate;
    * ``weight_norm`` clamped to ``(0, 10)``;
    * weight decay folded into ``adam_step`` BEFORE the trust ratio scales it;
    * ``trust_ratio = 1`` when either norm is zero.

    A generic LAMB gets at least the first three wrong, and neither `torch_optimizer`
    nor `apex` is installed in this environment -- hence a port with a parity test
    rather than a dependency.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))
        self.adam = adam

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step_size = group['lr']            # no bias correction, by design

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])
                adam_norm = adam_step.pow(2).sum().sqrt()
                trust_ratio = (1 if weight_norm == 0 or adam_norm == 0
                               else weight_norm / adam_norm)
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1
                p.data.add_(adam_step, alpha=-step_size * trust_ratio)
        return loss


def encode_jsonl_to_pickle(model_path, input_file, output_pkl, *, is_query, max_len,
                           batch_size):
    """Encode a JSONL and write ``(embeddings, ids)`` -- the tuple the miner reads.

    A dedicated process rather than Tevatron's encode driver, which rebuilds a stock
    DenseModel and would drop the projection head. Writing the SAME pickle contract
    keeps `build_faiss_index`, `mine_from_index`, `publish_round` and `read_round`
    untouched, so `run_ance_data_gen.py` needs no paper branch at all.

    Dynamic padding: 512 is a cap. MS MARCO passages average ~75 word-pieces, so
    padding to the cap would multiply the encode cost several-fold for nothing.

    The caller runs this entry point in a child process. Its parent owns the hard
    timeout, so a blocked tokenizer, model load or CUDA call can be terminated rather
    than relying on an in-process deadline that never regains control.

    Allocates exactly one embedding array and never copies it. At MS MARCO scale that
    array is 27 GiB, so a second live copy is the difference between fitting the node
    and thrashing it.
    """
    import json
    import pickle
    import time
    import resource
    import numpy as np
    from transformers import AutoTokenizer

    id_key, text_key = ('query_id', 'query') if is_query else ('docid', 'text')

    # Count first, then fill in place. ONE 8.8M x 768 float32 array is 27 GiB and the
    # node has 128 GB, so the old shape -- accumulate per-batch arrays, concatenate
    # (two full copies live at once), then `.astype(np.float32)` on already-float32
    # data (a third) -- peaked near 81 GiB before FAISS made its own. The count is a
    # line scan with no JSON parsing, which is cheap even at 8.8M rows.
    started = time.monotonic()
    with open(input_file, encoding='utf-8') as handle:
        n_rows = sum(1 for line in handle if line.strip())

    counted = time.monotonic()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_ance_encoder(model_path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    loaded = time.monotonic()
    batch_times = {'tokenize_s': 0.0, 'forward_transfer_s': 0.0}

    embeddings = np.empty((n_rows, EMBED_DIM), dtype=np.float32)
    ids, pending = [], []

    def _flush():
        """Encode one batch straight into its slice of the preallocated array."""
        if len(ids) > n_rows:
            raise RuntimeError(
                f"{input_file} grew while it was being encoded: more than {n_rows} "
                f"row(s). The buffer was sized before the encode began, so the extra "
                f"rows have nowhere to land.")
        t0 = time.monotonic()
        batch = tokenizer(pending, padding=True, truncation=True, max_length=max_len,
                          return_tensors='pt').to(device)
        t1 = time.monotonic()
        with torch.no_grad():
            vecs = model(input_ids=batch['input_ids'],
                         attention_mask=batch['attention_mask']).float().cpu().numpy()
        embeddings[len(ids) - len(pending):len(ids)] = vecs
        batch_times['tokenize_s'] += t1 - t0
        batch_times['forward_transfer_s'] += time.monotonic() - t1
        pending.clear()

    # Ids and texts come from the SAME pass, so row i always belongs to id i however
    # the file behaves. The text itself is streamed, never materialized: the corpus is
    # several GB of Python strings and nothing after the encode needs it.
    with open(input_file, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(str(row[id_key]))
            pending.append(row[text_key])
            if len(pending) == batch_size:
                _flush()
    if pending:
        _flush()

    if len(ids) < n_rows:
        # Shrank mid-encode. Trailing rows were never written, and returning them
        # would hand the miner uninitialized memory as embeddings.
        raise RuntimeError(
            f"{input_file} shrank while it was being encoded: {len(ids)} row(s) read "
            f"of the {n_rows} counted. The embeddings would not match the ids.")

    output_pkl = Path(output_pkl)
    encoded = time.monotonic()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump((embeddings, ids), f)
    finished = time.monotonic()
    telemetry = dict(batch_times, count_s=counted-started, model_load_s=loaded-counted,
                     encode_s=encoded-loaded, pickle_write_s=finished-encoded,
                     total_s=finished-started, rows=len(ids), bytes=output_pkl.stat().st_size,
                     rows_per_second=len(ids)/max(encoded-loaded, 1e-9),
                     peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     torch_threads=torch.get_num_threads())
    from utils.helpers import atomic_write
    with atomic_write(Path(str(output_pkl) + '.timings.json')) as handle:
        json.dump(telemetry, handle, indent=2)
    print(f'[Encode] {json.dumps(telemetry)}', flush=True)
    return output_pkl


def main():
    parser = argparse.ArgumentParser(description="Encode JSONL with the paper ANCE head")
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_pkl', required=True)
    parser.add_argument('--max_len', required=True, type=int)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--is_query', action='store_true')
    args = parser.parse_args()
    encode_jsonl_to_pickle(
        args.model_path, args.input_file, args.output_pkl, is_query=args.is_query,
        max_len=args.max_len, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
