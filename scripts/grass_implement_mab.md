# CaseBandit [S8] — Implementation Spec

CASE-inspired challenger-set bandit for `train_with_ema_grass()`.
Reference: https://github.com/kiranpurohit/CASE (sample_efficient_method.pdf in repo root)

---

## What we're building and why

EMA-GRASS currently mines ALL 32 queries every batch (4 encode passes each) → ~17h/epoch.
With CaseBandit: mine only n_das=5 (challengers) per batch, rest use neg_cache → ~3–4h/epoch.

The speedup is batch-level and immediate from batch 1 — not epoch-dependent.
Every query already has a negative in the training mixture files → that's the initial cache.

**CASE mapping:**
- Arm = training query
- Pull (LLM call) = mine_ema_batch() for that query — 4 encode passes
- Reward = sigma = |s_cur − s_ema| — already computed in mine_ema_batch, no proxy needed
- Challenger set N_t = top-n_das queries selected per batch by UCB
- J_t = queries with stable low sigma → graduated, cached negative frozen, never re-mined

---

## Step 1: Add `import math` at top of train_grass.py

Check if already present; if not, add after `import numpy as np`.

---

## Step 2: Add `CaseBandit` class to train_grass.py

Add immediately before `mine_ema_batch` function (around line 76).

```python
class CaseBandit:
    """
    [S8] CASE-style challenger-set bandit for EMA mining.
    Each batch: selects n_das queries (challengers) for full EMA mining.
    All other queries use neg_cache (initially mixture negatives).
    Reward: sigma = |s_cur - s_ema| — already computed in mine_ema_batch.
    Feature: [mean_sigma, 1.0] — updated via Sherman-Morrison as observations arrive.
    Cold start: UCB=inf for unseen queries → random n_das selection → fast calibration.
    J_t: queries with stable low sigma graduate → cached neg frozen, never re-mined.
    """
    def __init__(self, n_das=5, alpha=1.0, lambda_reg=1.0, epsilon=0.05, min_pulls=2):
        self.n_das     = n_das
        self.alpha     = alpha
        self.epsilon   = epsilon
        self.min_pulls = min_pulls

        d = 2  # feature: [mean_sigma, 1.0]
        self.V_inv  = (1.0 / lambda_reg) * np.eye(d)
        self.b      = np.zeros(d)
        self.theta  = np.zeros(d)

        self.n_pulls    = {}   # qid → int
        self.mean_sigma = {}   # qid → float (running mean of observed sigma)
        self.J_t        = set()  # graduated query IDs — never re-mined

    @staticmethod
    def feat(sigma_val):
        return np.array([float(sigma_val), 1.0])

    def ucb(self, qid):
        """UCB score for a query. inf for unseen queries."""
        if qid not in self.mean_sigma:
            return float('inf')
        x     = self.feat(self.mean_sigma[qid])
        mu    = float(self.theta @ x)
        bonus = self.alpha * math.sqrt(max(0.0, float(x @ self.V_inv @ x)))
        return mu + bonus

    def select(self, batch_ids):
        """Return set of top-n_das query IDs by UCB, excluding J_t members."""
        active = [qid for qid in batch_ids if qid not in self.J_t]
        if not active:
            return set()
        ranked = sorted(active, key=self.ucb, reverse=True)
        return set(ranked[:self.n_das])

    def update(self, qid, sigma_observed):
        """
        Update running mean, Sherman-Morrison V_inv, and check J_t graduation.
        Call once per mined query after observing its sigma from mine_ema_batch.
        """
        if qid not in self.n_pulls:
            self.n_pulls[qid]    = 0
            self.mean_sigma[qid] = 0.0

        self.n_pulls[qid] += 1
        n = self.n_pulls[qid]
        self.mean_sigma[qid] += (sigma_observed - self.mean_sigma[qid]) / n

        # Sherman-Morrison rank-1 update of V_inv
        x  = self.feat(sigma_observed)
        Vx = self.V_inv @ x
        self.V_inv -= np.outer(Vx, Vx) / (1.0 + float(x @ Vx))
        self.b     += sigma_observed * x
        self.theta  = self.V_inv @ self.b

        # J_t graduation: gap-index B(q, worst_in_J_t) <= epsilon
        if n >= self.min_pulls and self.J_t:
            worst = max(self.J_t, key=lambda q: self.mean_sigma.get(q, 0.0))
            x_q   = self.feat(self.mean_sigma[qid])
            x_w   = self.feat(self.mean_sigma.get(worst, 0.0))
            conf  = self.alpha * (
                math.sqrt(max(0.0, float(x_q @ self.V_inv @ x_q))) +
                math.sqrt(max(0.0, float(x_w @ self.V_inv @ x_w)))
            )
            gap = self.mean_sigma[qid] - self.mean_sigma.get(worst, 0.0)
            if (gap + conf) <= self.epsilon:
                self.J_t.add(qid)
        elif n >= self.min_pulls:
            self.J_t.add(qid)  # bootstrap J_t with first graduated queries
```

---

## Step 3: Modify `mine_ema_batch()` to also return sigma_scores

Current return: `return mined`
New return: `return mined, sigma_scores`

Add `sigma_scores = {}` at the top of the function body.
After `mined[qid] = [cands[k] for k in top_m]` (the line that sets the selected negative),
add: `sigma_scores[qid] = float(sigma[top_m[0]])`

Full diff for the return section (find the existing return and the loop that precedes it):

```python
# Add at top of mine_ema_batch function body (after the cfg unpacking lines):
sigma_scores = {}

# Add after: mined[qid] = [cands[k] for k in top_m]
sigma_scores[qid] = float(sigma[top_m[0]])

# Change final return from:
return mined
# To:
return mined, sigma_scores
```

---

## Step 4: Modify `train_with_ema_grass()`

### 4a. Add new config reads (after existing cfg unpacking, around line 181)

```python
n_das       = cfg.get('mab_n_das', batch_size)   # default = batch_size (disabled)
mab_alpha   = cfg.get('mab_alpha', 1.0)
mab_epsilon = cfg.get('mab_epsilon', 0.05)
mab_min_p   = cfg.get('mab_min_pulls', 2)
use_mab     = n_das < batch_size
```

### 4b. After loading train_items (before the epoch loop, around line 227)

Pre-populate neg_cache from mixture files so every query has a fallback from batch 1:

```python
# [S8] Pre-populate neg_cache from mixture negatives — every query has a fallback
# from batch 1 so no cold-start mining of all queries is needed.
neg_cache = {}  # qid → neg_docid
for f_path in sorted(mix_dir.glob("*.jsonl")):
    if f_path.name.startswith('.'): continue
    with open(f_path) as f:
        for line in f:
            d   = json.loads(line)
            qid = str(d['query_id'])
            negs = d.get('negative_passages', [])
            if negs and qid not in neg_cache:
                neg_cache[qid] = negs[0]['docid']

bandit = CaseBandit(n_das=n_das, alpha=mab_alpha,
                     epsilon=mab_epsilon, min_pulls=mab_min_p) if use_mab else None
if use_mab:
    print(f"  [S8] CaseBandit enabled: n_das={n_das}/{batch_size} per batch", flush=True)
```

### 4c. Replace the existing `mine_ema_batch` call (around line 245)

Find this block:
```python
# 1. Mine negatives per batch — all under no_grad
mined = mine_ema_batch(
    model, ema_model, tokenizer, batch_items,
    stale_idx, stale_embs, c_id_to_idx, c_ids,
    corpus_lookup, qrels_dict, cfg, config, device
)
```

Replace with:
```python
# 1. Mine negatives — [S8] challengers only (n_das queries), rest use neg_cache
batch_ids = [item['query_id'] for item in batch_items]

if bandit is not None:
    mine_set = bandit.select(batch_ids)
else:
    mine_set = set(batch_ids)

mine_items = [it for it in batch_items if it['query_id'] in mine_set]

mined = {}
if mine_items:
    mined_sub, sigma_scores = mine_ema_batch(
        model, ema_model, tokenizer, mine_items,
        stale_idx, stale_embs, c_id_to_idx, c_ids,
        corpus_lookup, qrels_dict, cfg, config, device
    )
    for it in mine_items:
        qid = it['query_id']
        if mined_sub.get(qid):
            mined[qid]     = mined_sub[qid]
            neg_cache[qid] = mined_sub[qid][0]   # refresh cache
            if bandit is not None:
                bandit.update(qid, float(sigma_scores.get(qid, 0.0)))

# Non-mined queries fall back to neg_cache
for it in batch_items:
    qid = it['query_id']
    if qid not in mined and qid in neg_cache:
        mined[qid] = [neg_cache[qid]]
```

### 4d. Update logging (inside the `if global_step % logging_steps == 0` block)

Add after the existing ETA print:
```python
if bandit is not None:
    jt_size  = len(bandit.J_t)
    seen     = len(bandit.n_pulls)
    skip_pct = f"{jt_size/max(1,seen):.1%}"
    print(f"  [S8] J_t={jt_size} graduated | skip ratio={skip_pct}", flush=True)
```

---

## Step 5: Config changes (config/config.yaml, under training.grass)

Add after the existing EMA-specific section:

```yaml
# --- CaseBandit MAB [S8] (EMA mode only) ---
mab_n_das:    5      # challengers mined per batch (out of ema_batch_size=32)
mab_alpha:    1.0    # LinUCB exploration constant
mab_epsilon:  0.05   # gap-index threshold for J_t graduation
mab_min_pulls: 2     # min mines before J_t eligibility
```

---

## Step 6: New tests in grass_test.py (tag [S8])

Add to the bottom of `grass_test.py`, before the `suite` list:

```python
# -----------------------------------------------------------------------
# S8 — CaseBandit correctness
# -----------------------------------------------------------------------

def test_s8_unseen_queries_always_selected():
    """Unseen queries (UCB=inf) must always rank above any seen query."""
    bandit = _mod.CaseBandit(n_das=2, alpha=1.0)
    # Give query A a high observed sigma
    bandit.update("qA", 0.9)
    bandit.update("qA", 0.9)
    # qB is unseen — must be selected over qA despite qA having higher sigma
    selected = bandit.select(["qA", "qB", "qC"])
    assert "qB" in selected and "qC" in selected, \
        f"Unseen queries not prioritised: selected={selected}"


def test_s8_jt_queries_never_selected():
    """select() must never return a query already in J_t."""
    bandit = _mod.CaseBandit(n_das=3, alpha=0.0, epsilon=1.0, min_pulls=1)
    # Force qA into J_t by giving it a low sigma observation
    bandit.J_t.add("qA")
    selected = bandit.select(["qA", "qB", "qC", "qD"])
    assert "qA" not in selected, f"J_t query was selected: {selected}"


def test_s8_low_sigma_graduates_to_jt():
    """After min_pulls updates with sigma near 0, query must enter J_t."""
    bandit = _mod.CaseBandit(n_das=2, alpha=0.0, epsilon=0.5, min_pulls=2)
    # Seed J_t with a high-sigma query so graduation check has a reference
    bandit.update("qRef", 0.8)
    bandit.J_t.add("qRef")
    bandit.mean_sigma["qRef"] = 0.8
    # Now observe very low sigma for qNew twice
    bandit.update("qNew", 0.01)
    bandit.update("qNew", 0.01)
    assert "qNew" in bandit.J_t, \
        f"Low-sigma query did not graduate to J_t after min_pulls"
```

Add to `suite` list in `__main__`:
```python
("S8  unseen queries always selected over seen",    test_s8_unseen_queries_always_selected),
("S8  J_t queries never returned by select()",      test_s8_jt_queries_never_selected),
("S8  low-sigma query graduates to J_t",            test_s8_low_sigma_graduates_to_jt),
```

Update the expected count message from `12/12` to `15/15`.

---

## Verification checklist

```bash
# 1. Syntax check
python -c "import ast; ast.parse(open('scripts/train_grass.py').read()); print('OK')"

# 2. All tests pass
python scripts/grass_test.py
# Expect: 15/15 passed

# 3. Debug smoke test — confirm n_das=5 mines per batch from batch 1
python scripts/train_grass.py --debug --mode ema
# Look for: "[S8] CaseBandit enabled: n_das=5/32 per batch"
# Look for: "[S8] J_t=0 graduated | skip ratio=0.0%" early on, growing later

# 4. Confirm neg_cache pre-populated (add temp print after init):
# print(f"neg_cache size: {len(neg_cache)}")  → expect ~330K
```

---

## What NOT to change

- `grass_sampler()` (mc_dropout) — untouched
- `encode_batch()`, `encode_batch_train()` — untouched
- `_shortlist_batch()` — untouched
- `main()` — untouched
- All S1–S7 tags — untouched
