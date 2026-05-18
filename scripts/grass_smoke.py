"""
GRASS async smoke test — runs locally on CPU, no GPU or internet needed.

Creates a tiny workspace (20 queries, 50 passages, dim=16) and runs the
miner and trainer as threads with a mock model. Verifies:
  1. Miner produces ready_N markers (IPC round-trip works)
  2. Trainer writes optimizer.pt last (validity gate works)
  3. Miner detects new checkpoint (is_valid_checkpoint gate fires)
  4. Trainer applies miner updates (neg_cache updates logged)
  5. Both threads exit cleanly (no deadlock, no crash)

Run: python scripts/grass_smoke.py
"""
import gc
import json
import os
import pickle
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

# Must be set before importing faiss or torch (OpenMP duplicate-lib on macOS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import faiss

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.helpers import is_valid_checkpoint, get_latest_marker_no
import importlib, sys as _sys
_train_spec = importlib.util.spec_from_file_location(
    '_smoke_train', Path(__file__).parent / 'run_grass_train.py'
)
_train_mod = importlib.util.module_from_spec(_train_spec)
_sys.modules.setdefault('models', __import__('unittest.mock', fromlist=['MagicMock']).MagicMock())
_sys.modules.setdefault('models.temperature_scaled_loss', __import__('unittest.mock', fromlist=['MagicMock']).MagicMock())
_train_spec.loader.exec_module(_train_mod)
_apply_pending_neg_updates = _train_mod._apply_pending_neg_updates

# ── Tiny mock model (no HuggingFace download) ──────────────────────────────

DIM = 16


class _Out:
    def __init__(self, x):
        self.last_hidden_state = x


class TinyModel(nn.Module):
    """Random-weight model shaped like a transformer. Has Dropout for MC diversity."""
    def __init__(self):
        super().__init__()
        self.proj    = nn.Linear(4, DIM)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        B = input_ids.shape[0]
        x = self.dropout(self.proj(torch.randn(B, 4, 4)))
        return _Out(x)

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "pytorch_model.bin")

    def parameters(self):
        return super().parameters()

    def modules(self):
        return super().modules()


class TinyTokenizer:
    def __call__(self, texts, padding=True, truncation=True,
                 max_length=64, return_tensors='pt'):
        B = len(texts)

        class _Enc(dict):
            def to(self, device): return self

        return _Enc({
            'input_ids':      torch.zeros(B, 4, dtype=torch.long),
            'attention_mask': torch.ones(B, 4, dtype=torch.long),
        })

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


# ── Workspace builder ──────────────────────────────────────────────────────

def _build_workspace(root):
    """Create all synthetic files the miner and trainer expect."""
    root = Path(root)
    n_pass, n_q = 50, 20

    # Corpus
    corpus = [{'docid': f"doc{i}", 'text': f"passage text {i}"} for i in range(n_pass)]
    corpus_file = root / "corpus.jsonl"
    with open(corpus_file, 'w') as f:
        for d in corpus:
            f.write(json.dumps(d) + '\n')

    # Queries
    queries = [{'query_id': f"q{i}", 'query': f"query text {i}"} for i in range(n_q)]
    query_file = root / "queries.jsonl"
    with open(query_file, 'w') as f:
        for d in queries:
            f.write(json.dumps(d) + '\n')

    # Qrels (each query → doc with same index as positive)
    qrels_file = root / "qrels.txt"
    with open(qrels_file, 'w') as f:
        for i in range(n_q):
            f.write(f"q{i} 0 doc{i} 1\n")

    # Stale FAISS pickle
    rng    = np.random.default_rng(0)
    embs   = rng.standard_normal((n_pass, DIM)).astype(np.float32)
    embs  /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    c_ids  = [f"doc{i}" for i in range(n_pass)]
    stale_pkl = root / "stale.pkl"
    with open(stale_pkl, 'wb') as f:
        pickle.dump((embs, c_ids), f)

    # Mixture JSONL (trainer loads from training_mixture/)
    mix_dir = root / "training_mixture"
    mix_dir.mkdir()
    mix_file = mix_dir / "train_smoke.jsonl"
    with open(mix_file, 'w') as f:
        for i in range(n_q):
            rec = {
                'query_id': f"q{i}",
                'query':    f"query text {i}",
                'positive_passages':  [{'docid': f"doc{i}",        'text': f"passage text {i}"}],
                'negative_passages':  [{'docid': f"doc{(i+1)%n_pass}", 'text': f"passage text {(i+1)%n_pass}"}],
            }
            f.write(json.dumps(rec) + '\n')

    corpus_lookup = {d['docid']: d['text'] for d in corpus}
    qrels_dict    = {f"q{i}": {f"doc{i}"} for i in range(n_q)}
    query_ids     = [f"q{i}" for i in range(n_q)]
    qid_to_text   = {f"q{i}": f"query text {i}" for i in range(n_q)}

    return (corpus_file, query_file, qrels_file, stale_pkl,
            corpus_lookup, qrels_dict, query_ids, qid_to_text, mix_dir)


# ── Miner stub ──────────────────────────────────────────────────────────────

def _run_miner(output_model_dir, neg_update_dir, stale_embs, c_ids,
               query_ids, qid_to_text, corpus_lookup, qrels_dict,
               stop_evt, results):
    """
    Minimal miner thread. Uses numpy dot-product ANN (no FAISS in thread —
    FAISS + PyTorch OpenMP conflict on macOS; FAISS correctness is tested in S7).
    Tests: IPC write, checkpoint gate, bandit select_global.
    """
    from transformers.trainer_utils import get_last_checkpoint
    from utils.helpers import encode_batch, _shortlist_batch
    from utils.bandit import CaseBandit

    try:
        c_id_to_idx = {did: i for i, did in enumerate(c_ids)}
        model     = TinyModel()
        tokenizer = TinyTokenizer()
        device    = torch.device('cpu')
        model.train()

        bandit = CaseBandit(n_das=3, epsilon=0.2)
        bandit.init_all_queries(query_ids)

        update_num = 1
        last_ckpt  = None
        n_das, T, P, L, m, lam = 3, 2, 10, 5, 1, 2.0

        print("[Miner] Started", flush=True)

        while not stop_evt.is_set():
            # Checkpoint gate
            ckpt = get_last_checkpoint(str(output_model_dir))
            if ckpt and ckpt != last_ckpt and is_valid_checkpoint(ckpt):
                last_ckpt = ckpt
                results['miner_saw_checkpoint'] = True
                print(f"[Miner] Checkpoint gate passed: {Path(ckpt).name}", flush=True)

            # Select queries
            selected = bandit.select_global(n_das=n_das, epsilon=0.2)
            selected = [q for q in selected if q in qid_to_text]
            if not selected:
                time.sleep(0.1)
                continue

            texts = [qid_to_text[q] for q in selected]

            # Deterministic encode for ANN (numpy brute-force, no FAISS in thread)
            model.eval()
            q_det = encode_batch(model, tokenizer, texts, device, 64, 8)
            model.train()

            # Brute-force top-P via numpy instead of FAISS
            scores  = stale_embs @ q_det.T           # (n_pass, n_q)
            indices = np.argsort(-scores, axis=0)[:P].T  # (n_q, P)

            # Shortlist
            sl, sl_ids, sl_texts, sl_to_idx, _ = _shortlist_batch(
                selected, indices, q_det, qrels_dict,
                c_ids, c_id_to_idx, stale_embs, corpus_lookup, P, L
            )

            # T MC encodes (vectorized)
            q_flat  = encode_batch(model, tokenizer, texts * T, device, 64, 8)
            q_stack = q_flat.reshape(T, len(texts), -1)
            c_flat  = encode_batch(model, tokenizer, sl_texts * T, device, 64, 8) if sl_texts else None
            c_stack = c_flat.reshape(T, len(sl_texts), -1) if c_flat is not None else None

            mined  = {}
            sigmas = []
            for i, qid in enumerate(selected):
                cands = sl.get(qid, [])
                if not cands or c_stack is None:
                    continue
                cidxs = [sl_to_idx[d] for d in cands]
                sims  = np.einsum('td,tnd->tn', q_stack[:, i, :], c_stack[:, cidxs, :])
                s_hat = sims.mean(axis=0)
                sigma = sims.std(axis=0)
                g     = s_hat + lam * sigma
                top   = np.argsort(g)[::-1][:m]
                mined[qid] = [cands[k] for k in top]
                top_sigma = float(sigma[top[0]]) if len(top) else 0.0
                sigmas.append(top_sigma)
                bandit.update(qid, top_sigma)

            if mined:
                jpath = neg_update_dir / f"update_{update_num}.jsonl"
                with open(jpath, 'w') as f:
                    for qid, negs in mined.items():
                        if negs:
                            f.write(json.dumps({'query_id': qid,
                                                'neg_docid': negs[0]}) + '\n')
                (neg_update_dir / f"ready_{update_num}").write_text(str(update_num))
                results['miner_updates'] = update_num
                if sigmas:
                    arr = np.array(sigmas)
                    print(
                        f"[Miner] #{update_num} | queries={len(mined)} | "
                        f"σ mean={arr.mean():.4f} std={arr.std():.4f} "
                        f"min={arr.min():.4f} max={arr.max():.4f} | "
                        f"exploit=2 explore=1 | "
                        f"J_t={len(bandit.J_t)} unseen={len(bandit.unseen)}",
                        flush=True,
                    )
                update_num += 1

            time.sleep(0.1)

        results['miner_clean_exit'] = True
        print("[Miner] Stopped cleanly", flush=True)

    except Exception as e:
        results['miner_error'] = str(e)
        import traceback; traceback.print_exc()


# ── Trainer stub ────────────────────────────────────────────────────────────

def _run_trainer(output_model_dir, neg_update_dir, corpus_lookup, mix_dir,
                 stop_evt, results):
    """
    Minimal trainer thread. Tests: neg_cache init, neg update polling,
    checkpoint written with optimizer.pt last.
    """
    from torch.optim import SGD

    try:
        model     = TinyModel()
        tokenizer = TinyTokenizer()
        device    = torch.device('cpu')
        optimizer = SGD(model.parameters(), lr=1e-3)
        model.train()

        # Load training data
        train_items = []
        for f in sorted(mix_dir.glob("*.jsonl")):
            with open(f) as fp:
                for line in fp:
                    d = json.loads(line)
                    pos  = d.get('positive_passages', [])
                    negs = d.get('negative_passages', [])
                    if pos:
                        train_items.append({
                            'query_id':  str(d['query_id']),
                            'query':     d['query'],
                            'pos_docid': pos[0]['docid'],
                            'neg_docid': negs[0]['docid'] if negs else None,
                        })

        neg_cache = {it['query_id']: it['neg_docid']
                     for it in train_items if it['neg_docid']}
        last_update_no = 0
        save_steps     = 5
        neg_poll_steps = 3

        print(f"[Trainer] {len(train_items)} examples, starting loop", flush=True)

        for step in range(25):
            if stop_evt.is_set():
                break

            # Poll for neg updates
            if step % neg_poll_steps == 0:
                last_update_no, n_applied = _apply_pending_neg_updates(
                    neg_update_dir, neg_cache, last_update_no
                )
                if n_applied > 0:
                    results['trainer_applied_updates'] = True
                    print(f"[Trainer] step={step}: applied {n_applied} neg updates", flush=True)

            # Fake forward/backward
            item = train_items[step % len(train_items)]
            q_enc  = tokenizer([item['query']])
            loss   = model(input_ids=q_enc['input_ids']).last_hidden_state.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save checkpoint (optimizer.pt last — validity gate)
            if (step + 1) % save_steps == 0:
                ckpt = output_model_dir / f"checkpoint-{step+1}"
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt))
                tokenizer.save_pretrained(str(ckpt))
                torch.save({'step': step}, ckpt / "scheduler.pt")
                torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")  # LAST
                results['trainer_saved_checkpoint'] = str(ckpt)
                print(f"[Trainer] Checkpoint: {ckpt.name}", flush=True)

            time.sleep(0.05)

        results['trainer_clean_exit'] = True
        print("[Trainer] Done", flush=True)
        stop_evt.set()

    except Exception as e:
        results['trainer_error'] = str(e)
        stop_evt.set()
        import traceback; traceback.print_exc()


# ── Main ────────────────────────────────────────────────────────────────────

def run_smoke():
    print("\nGRASS Async Smoke Test")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        root             = Path(tmpdir)
        output_model_dir = root / "model"
        neg_update_dir   = root / "neg_updates"
        output_model_dir.mkdir()
        neg_update_dir.mkdir()

        (corpus_file, query_file, qrels_file, stale_pkl,
         corpus_lookup, qrels_dict, query_ids, qid_to_text, mix_dir) = _build_workspace(root)

        with open(stale_pkl, 'rb') as _f:
            stale_embs, c_ids = pickle.load(_f)

        results  = {}
        stop_evt = threading.Event()

        miner_t  = threading.Thread(
            target=_run_miner,
            args=(output_model_dir, neg_update_dir, stale_embs, c_ids,
                  query_ids, qid_to_text, corpus_lookup, qrels_dict,
                  stop_evt, results),
            daemon=True,
        )
        trainer_t = threading.Thread(
            target=_run_trainer,
            args=(output_model_dir, neg_update_dir, corpus_lookup,
                  mix_dir, stop_evt, results),
            daemon=True,
        )

        miner_t.start()
        trainer_t.start()

        # Wait for trainer to finish (it stops after 25 steps, sets stop_evt)
        trainer_t.join(timeout=30)
        miner_t.join(timeout=5)

        print("\n" + "=" * 50)
        print("Results:")

        checks = [
            ("Trainer completed without error",
             'trainer_error' not in results and results.get('trainer_clean_exit')),
            ("Miner completed without error",
             'miner_error' not in results and results.get('miner_clean_exit')),
            ("Miner produced ≥1 neg update",
             results.get('miner_updates', 0) >= 1),
            ("Trainer saved checkpoint with optimizer.pt last",
             bool(results.get('trainer_saved_checkpoint')) and
             is_valid_checkpoint(results['trainer_saved_checkpoint'])),
            ("Miner detected trainer checkpoint",
             results.get('miner_saw_checkpoint', False)),
            ("Trainer applied ≥1 miner neg update",
             results.get('trainer_applied_updates', False)),
        ]

        passed = 0
        for name, ok in checks:
            status = "✅ PASS" if ok else "❌ FAIL"
            print(f"  {status}  {name}")
            if ok:
                passed += 1

        if 'miner_error' in results:
            print(f"\n  [Miner error] {results['miner_error']}")
        if 'trainer_error' in results:
            print(f"\n  [Trainer error] {results['trainer_error']}")

        print("=" * 50)
        total = len(checks)
        print(f"  {passed}/{total} passed", end="  ")
        if passed == total:
            print("— wiring looks good. Submit to cluster.")
        else:
            print("— fix failures before cluster.")
        print("=" * 50)
        return passed == total


if __name__ == "__main__":
    ok = run_smoke()
    sys.exit(0 if ok else 1)
