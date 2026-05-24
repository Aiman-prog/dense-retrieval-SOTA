import heapq
import math
import random


class EpsilonGreedyBandit:
    """
    Minimal ε-greedy bandit with EMA-updated gap-index estimates.

    Selection (SelectGlobal): exploit top-(1-ε)·n distinct queries by mean_g
                              + uniform random ε·n from queries not in exploit.
    Update: EMA with constant step-size α — tracks current-model g under
            non-stationary reward (Sutton & Barto §2.5).

    Heap uses lazy deletion: every update() pushes a fresh (priority, version, qid)
    entry; stale entries are skipped on pop via the version counter. No J_t, no
    cooldown, no stale_cycles, no annealing.

    Used by the sequential-bandit pipeline (scripts/run_grass_seq_bandit.py) and
    the async v2 pipeline (scripts/run_grass_async_v2_miner.py).
    """
    def __init__(self, epsilon=0.3, alpha=0.5):
        self.epsilon   = epsilon
        self.alpha     = alpha
        self.mean_g    = {}   # qid -> running EMA estimate
        self.version   = {}   # qid -> int (fresh-entry counter for lazy heap)
        self.heap      = []   # min-heap of (-mean_g, version, qid)
        self._all_qids = []   # full query pool, for explore sampling

    def init_query_pool(self, all_query_ids):
        """Register the full query pool. Call once before any select_global."""
        self._all_qids = list(all_query_ids)
        for qid in self._all_qids:
            self.mean_g.setdefault(qid, 0.0)
            self.version.setdefault(qid, 0)

    def update(self, qid, g_observed):
        """EMA-update mean_g[qid] and push a fresh heap entry."""
        old = self.mean_g.get(qid, 0.0)
        self.mean_g[qid] = self.alpha * g_observed + (1.0 - self.alpha) * old
        ver = self.version.get(qid, 0) + 1
        self.version[qid] = ver
        heapq.heappush(self.heap, (-self.mean_g[qid], ver, qid))

    def select_global(self, n):
        """Return n distinct query IDs: top (1-ε)·n by mean_g + random sample for the rest.

        If exploit cannot meet its budget (e.g. heap empty before init pass runs, or
        heap exhausted), explore fills the gap so the total returned is min(n, |D|).
        """
        n_exp = int((1.0 - self.epsilon) * n)

        # Exploit: pop top-n_exp distinct fresh queries (skip stale heap entries)
        exploit = []
        seen    = set()
        while len(exploit) < n_exp and self.heap:
            _, ver, qid = heapq.heappop(self.heap)
            if self.version.get(qid, 0) == ver and qid not in seen:
                exploit.append(qid)
                seen.add(qid)

        # Explore: fill remainder of n with random sample from queries not in exploit
        n_random = n - len(exploit)
        explore_pool = [q for q in self._all_qids if q not in seen]
        explore = random.sample(explore_pool, min(n_random, len(explore_pool)))

        return exploit + explore


class CaseLiteBandit:
    """
    Bucket-UCB challenger allocator + incumbent tracker for CASE-Lite.

    See grass-report.tex §6 (CASE-Lite Challenger Sampling for GRASS).

    Arms are cheap-rank buckets pooled globally across queries. Per query,
    a fixed K-budget is split into 1 incumbent + (K-1) challengers, with
    the challenger-slot allocation chosen by Bucket-UCB:
        UCB_b = mu_b + beta * sqrt(log(1 + N_total) / (1 + N_b))

    Round 1 uses the static initial_slots prior (per §6.4). Round 2+ uses
    proportional allocation by UCB. Bucket reward statistics are updated
    once per round (two-level: within-round mean -> EMA across rounds, §6.6).

    incumbent[qid] tracks the current best-known negative per query and is
    re-scored every round it competes (§6.3).
    """
    def __init__(self, bucket_boundaries, initial_slots,
                 alpha_b=0.5, beta=0.5, gamma=0.05, tau=0.0, lambda_val=1.0):
        assert len(bucket_boundaries) == len(initial_slots), \
            "bucket_boundaries and initial_slots must align"
        self.bucket_boundaries = list(bucket_boundaries)  # e.g. [5, 10, 25] (upper bounds, 1-indexed)
        self.initial_slots     = list(initial_slots)      # e.g. [3, 1, 1] (sums to K-1)
        self.alpha_b           = alpha_b
        self.beta              = beta
        self.gamma             = gamma
        self.tau               = tau
        self.lambda_val        = lambda_val
        self.n_buckets         = len(bucket_boundaries)
        self.N_b               = [0]   * self.n_buckets
        self.mu_b              = [0.0] * self.n_buckets
        self.incumbent         = {}  # qid -> docid

    def bucket_of(self, cheap_rank):
        """Map a 1-indexed cheap rank to a bucket index. Ranks beyond the last
        boundary clamp to the last bucket."""
        for b, ub in enumerate(self.bucket_boundaries):
            if cheap_rank <= ub:
                return b
        return self.n_buckets - 1

    def allocate_slots(self, K, round_idx):
        """Return a list of length n_buckets summing to K-1 challenger slots.

        Round 1: use static initial_slots prior (mu_b are uninformative).
        Round 2+: proportional allocation by Bucket-UCB priority, with
                  largest-remainder rounding to preserve the K-1 total.
        """
        budget = K - 1
        if budget <= 0:
            return [0] * self.n_buckets
        if round_idx <= 1:
            # If initial_slots doesn't sum to budget, scale; defensive but normally exact.
            s = sum(self.initial_slots)
            return list(self.initial_slots) if s == budget else \
                self._proportional([float(x) for x in self.initial_slots], budget)
        N_total = sum(self.N_b)
        ucb = [self.mu_b[b] + self.beta * math.sqrt(math.log(1 + N_total) / (1 + self.N_b[b]))
               for b in range(self.n_buckets)]
        # UCB priorities can be negative if mu_b is negative; shift to non-negative for proportional split.
        floor = min(ucb)
        weights = [u - floor + 1e-9 for u in ucb]
        return self._proportional(weights, budget)

    @staticmethod
    def _proportional(weights, budget):
        """Largest-remainder allocation: integer slots per bucket, summing to budget."""
        total_w = sum(weights)
        if total_w <= 0:
            # Degenerate: spread budget round-robin starting at bucket 0.
            slots = [budget // len(weights)] * len(weights)
            for i in range(budget % len(weights)):
                slots[i] += 1
            return slots
        raw     = [budget * w / total_w for w in weights]
        floors  = [int(x) for x in raw]
        rema    = [x - f for x, f in zip(raw, floors)]
        slots   = list(floors)
        leftover = budget - sum(floors)
        # Distribute leftover to the largest remainders.
        order = sorted(range(len(weights)), key=lambda i: rema[i], reverse=True)
        for i in order[:leftover]:
            slots[i] += 1
        return slots

    def update_round(self, round_rewards):
        """Apply per-round EMA update once. round_rewards: {b: [r, ...]}.

        For each bucket with non-empty observations:
            r_bar  = mean(round_rewards[b])
            mu_b   <- alpha_b * r_bar + (1 - alpha_b) * mu_b
            N_b    += len(round_rewards[b])
        Buckets with no observations this round are left unchanged.
        """
        for b, rewards in round_rewards.items():
            if not rewards:
                continue
            r_bar = sum(rewards) / len(rewards)
            self.mu_b[b] = self.alpha_b * r_bar + (1.0 - self.alpha_b) * self.mu_b[b]
            self.N_b[b] += len(rewards)
