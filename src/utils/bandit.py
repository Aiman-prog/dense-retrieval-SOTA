import heapq
import random


class CaseBandit:
    """
    Epsilon-greedy bandit over a mean-σ max-heap for global query selection.

    Fixes the batch-level UCB bug: select_global() picks from the full query pool
    (initialized via init_all_queries), not just the current training batch.

    Exploitation (1-ε fraction): pop top mean-σ queries from the heap.
    Exploration (ε fraction): random sample from the unseen set.
    J_t: queries with consistently low σ graduate and are excluded permanently.

    The legacy select(batch_ids) method is preserved for run_grass_ema.py compatibility.
    """
    def __init__(self, n_das=5, alpha=1.0, epsilon=0.2, min_pulls=2):
        self.n_das      = n_das
        self.epsilon    = epsilon
        self.min_pulls  = min_pulls
        self.mean_sigma = {}    # qid → float (running mean of observed σ)
        self.n_pulls    = {}    # qid → int
        self.J_t        = set() # graduated query IDs — never re-mined
        self.version    = {}    # qid → int (lazy-deletion counter for heap)
        self.unseen     = set() # queries not yet observed
        self.heap       = []    # max-heap entries: (-sigma, version, qid)

    def init_all_queries(self, all_query_ids):
        """Initialize heap with all query IDs at σ=0. O(n) via heapify."""
        self.unseen = set(all_query_ids)
        self.heap   = [(-0.0, 0, qid) for qid in all_query_ids]
        heapq.heapify(self.heap)

    def _heap_pop_top(self, n):
        """Pop up to n queries by highest mean-σ, skipping stale or graduated entries.
        Popped queries are NOT re-pushed — they re-enter only when update() is called.
        This prevents the same top-K queries from monopolising every mining cycle."""
        results = []
        while len(results) < n and self.heap:
            neg_sigma, ver, qid = heapq.heappop(self.heap)
            if self.version.get(qid, 0) == ver and qid not in self.J_t:
                results.append(qid)
        return results

    def select_global(self, n_das=None, epsilon=None):
        """Return n_das query IDs: exploit top-σ heap + explore random unseen."""
        if n_das is None:
            n_das = self.n_das
        if epsilon is None:
            epsilon = self.epsilon
        n_exploit   = int(n_das * (1 - epsilon))
        n_explore   = n_das - n_exploit
        exploit_ids = self._heap_pop_top(n_exploit)
        explore_pool = list(self.unseen - self.J_t)
        explore_ids  = random.sample(explore_pool, min(n_explore, len(explore_pool)))
        return exploit_ids + explore_ids

    def update(self, qid, sigma_observed):
        """Update running mean-σ, push fresh heap entry, check J_t graduation."""
        n = self.n_pulls.get(qid, 0) + 1
        self.n_pulls[qid] = n
        old = self.mean_sigma.get(qid, 0.0)
        self.mean_sigma[qid] = old + (sigma_observed - old) / n
        self.unseen.discard(qid)
        ver = self.version.get(qid, 0) + 1
        self.version[qid] = ver
        heapq.heappush(self.heap, (-self.mean_sigma[qid], ver, qid))
        # J_t graduation: query graduates if its mean-σ is below the worst J_t member
        if n >= self.min_pulls:
            if not self.J_t:
                self.J_t.add(qid)
            else:
                worst = max(self.J_t, key=lambda q: self.mean_sigma.get(q, 0.0))
                if self.mean_sigma[qid] <= self.mean_sigma.get(worst, 0.0):
                    self.J_t.add(qid)

    def select(self, batch_ids):
        """Legacy batch-level select for run_grass_ema.py. Scores by mean-σ (∞ for unseen)."""
        active = [qid for qid in batch_ids if qid not in self.J_t]
        if not active:
            return set()
        ranked = sorted(active, key=lambda q: self.mean_sigma.get(q, float('inf')), reverse=True)
        return set(ranked[:self.n_das])
