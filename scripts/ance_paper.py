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

    In-process rather than through Tevatron's encode driver, which rebuilds a stock
    DenseModel and would drop the projection head. Writing the SAME pickle contract
    keeps `build_faiss_index`, `mine_from_index`, `publish_round` and `read_round`
    untouched, so `run_ance_data_gen.py` needs no paper branch at all.

    Dynamic padding: 512 is a cap. MS MARCO passages average ~75 word-pieces, so
    padding to the cap would multiply the encode cost several-fold for nothing.
    """
    import json
    import pickle
    import numpy as np
    from transformers import AutoTokenizer

    id_key, text_key = ('query_id', 'query') if is_query else ('docid', 'text')
    ids, texts = [], []
    with open(input_file, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                ids.append(str(row[id_key]))
                texts.append(row[text_key])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_ance_encoder(model_path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                              max_length=max_len, return_tensors='pt').to(device)
            out.append(model(input_ids=batch['input_ids'],
                             attention_mask=batch['attention_mask']).float().cpu().numpy())
    embeddings = (np.concatenate(out, axis=0) if out
                  else np.zeros((0, EMBED_DIM), dtype=np.float32))

    output_pkl = Path(output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump((embeddings.astype(np.float32), ids), f)
    return output_pkl
