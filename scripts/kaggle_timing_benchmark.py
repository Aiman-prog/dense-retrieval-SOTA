"""
BGE-M3 Mining Throughput Benchmark — Kaggle T4 / any single GPU
================================================================
Measures encode_batch time for the async GRASS miner at different n_das values.
No real data needed — uses dummy texts to isolate pure encoding cost.

Run as a Kaggle notebook: paste each cell block (marked # CELL N) into a new cell.
Total runtime: ~10-20 min on T4.
"""

# CELL 1 — install
# !pip install -q transformers

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — benchmark
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ── config (must match config/config.yaml) ───────────────────────────────────
MODEL        = "BAAI/bge-m3"
L            = 10     # shortlist size per query (reduced from 25 for coverage)
T            = 5      # MC-dropout passes
mc_batch_sz  = 512    # encode_batch sub-batch size
mc_dropout_p = 0.3    # dropout probability
q_max_len    = 256
p_max_len    = 128
N_WARMUP     = 1
N_MEASURE    = 3
N_DAS_VALUES = [100, 150, 200, 300]  # knee expected ~164 for L=10

# ── setup ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\nLoading BGE-M3 ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)

# MC-dropout: keep dropout active, set p=0.3
model.train()
for module in model.modules():
    if isinstance(module, torch.nn.Dropout):
        module.p = mc_dropout_p
print("Model loaded.\n")


def encode_batch(texts, max_len):
    all_embs = []
    for i in range(0, len(texts), mc_batch_sz):
        batch  = texts[i:i + mc_batch_sz]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_len, return_tensors='pt').to(device)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                              enabled=device.type == 'cuda'):
            out = model(**inputs)
        embs = out.last_hidden_state[:, 0, :]
        embs = torch.nn.functional.normalize(embs, dim=-1)
        all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


# Dummy texts — realistic lengths
dummy_query = (
    "What is the relationship between quantum entanglement and "
    "thermodynamic entropy in open systems?"
)
dummy_passage = (
    "Thermodynamics describes macroscopic systems in terms of state variables "
    "such as temperature, pressure, and entropy. Quantum mechanics provides a "
    "microscopic description of the same systems. "
) * 2   # ~128 tokens worth

# ── timing loop ──────────────────────────────────────────────────────────────
print(f"{'n_das':>6} | {'q_seqs':>7} | {'c_seqs':>7} | "
      f"{'cycle (s)':>10} | {'per-query (ms)':>15} | {'sub-batches':>12}")
print("-" * 72)

results = []
for n_das in N_DAS_VALUES:
    q_texts = [dummy_query]   * n_das
    c_texts = [dummy_passage] * (n_das * L)

    q_seqs = n_das * T          # query encode total sequences
    c_seqs = n_das * L * T      # candidate encode total sequences

    times = []
    for trial in range(N_WARMUP + N_MEASURE):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Mirrors the MC-dropout encode pattern in grass_sampler:
        encode_batch(q_texts * T, q_max_len)   # MC query encode
        encode_batch(c_texts * T, p_max_len)   # MC candidate encode

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if trial >= N_WARMUP:
            times.append(elapsed)

    mean_t    = float(np.mean(times))
    per_query = mean_t / n_das * 1000   # ms
    n_sub     = c_seqs / mc_batch_sz    # approximate sub-batches (candidate encode dominates)
    results.append((n_das, mean_t, per_query, n_sub))

    print(f"{n_das:>6} | {q_seqs:>7} | {c_seqs:>7} | "
          f"{mean_t:>10.2f} | {per_query:>15.1f} | {n_sub:>12.1f}")

# ── coverage estimates ────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("Coverage estimates (adjust A100_SPEEDUP if you know better)")
print("=" * 72)

# T4 → A100 speedup is roughly 4-5x for transformer inference.
# We compute coverage for a range to show sensitivity.
TOTAL_QUERIES = 367_000
SLURM_HOURS   = 7

best_per_query_ms = results[-1][2]   # largest n_das ≈ best GPU utilization

print(f"\n{'n_das':>6} | {'efficiency':>10} | {'A100 ×4 cov':>12} | {'A100 ×5 cov':>12}")
print("-" * 52)
for n_das, cycle_t, per_q_ms, _ in results:
    efficiency = best_per_query_ms / per_q_ms * 100
    for speedup, label in [(4, "×4"), (5, "×5")]:
        a100_cycle  = cycle_t / speedup
        n_cycles    = (SLURM_HOURS * 3600) / a100_cycle
        unique      = n_cycles * n_das * 0.9   # 90% unique (eps_start=0.8)
        coverage    = min(unique / TOTAL_QUERIES * 100, 100)
        if label == "×4":
            cov4 = coverage
        else:
            cov5 = coverage
    print(f"{n_das:>6} | {efficiency:>9.0f}% | {cov4:>11.0f}% | {cov5:>11.0f}%")

print("""
Notes:
  efficiency  = per-query time at this n_das vs best (largest n_das)
  A100 ×4/×5 = coverage estimate on DelftBlue A100 at 7h
  The 'knee'  = where efficiency stops improving → that's the n_das to use
  90% unique  = assumes eps_start=0.8 (80% explore arm, all unique in first sweep)
""")
