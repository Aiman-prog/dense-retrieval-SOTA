import heapq
import random


class CaseBandit:
    """
    Epsilon-greedy bandit over a mean-g max-heap for global query selection.
    Tracks mean gap-index g = ŝ + λσ per query — handles both BRIGHT (σ dominates)
    and MS MARCO (ŝ dominates) naturally.

    Exploitation (1-ε fraction): pop top mean-g queries from the heap.
    Exploration (ε fraction): random sample from the unseen/stale pool.
    J_t: queries with consistently low g graduate and are excluded permanently.
    ε annealing: linear decay from epsilon_start → epsilon over decay_cycles.
    Stale refresh: when explore pool empties, refill with queries not updated
                   in the last stale_cycles cycles — keeps exploration live
                   across multiple dataset sweeps.
    """
    def __init__(self, n_das=5, alpha=1.0, epsilon=0.2, min_pulls=2,
                 epsilon_start=None, decay_cycles=None, stale_cycles=None):
        self.n_das         = n_das
        self.epsilon       = epsilon            # final/base epsilon
        self.epsilon_start = epsilon_start if epsilon_start is not None else epsilon
        self.decay_cycles  = decay_cycles       # None = no annealing
        self.stale_cycles  = stale_cycles       # None = no stale refresh
        self.min_pulls     = min_pulls
        self.mean_score    = {}  # qid → running mean of observed g (or σ for legacy)
        self.n_pulls       = {}  # qid → int
        self.J_t           = set()  # graduated query IDs — never re-mined
        self.version       = {}  # qid → int (lazy-deletion counter for heap)
        self.unseen        = set()  # explore pool: unseen queries or stale ones
        self.heap          = []  # max-heap entries: (-score, version, qid)
        self.cycle         = 0   # incremented on each select_global call
        self.last_updated  = {}  # qid → cycle when last updated (for stale refresh)
        self._all_query_ids = []

    def init_all_queries(self, all_query_ids):
        """Initialize heap with all query IDs at score=0. O(n) via heapify."""
        self._all_query_ids = list(all_query_ids)
        self.unseen = set(all_query_ids)
        self.heap   = [(-0.0, 0, qid) for qid in all_query_ids]
        heapq.heapify(self.heap)

    def _current_epsilon(self):
        """Linear decay from epsilon_start to epsilon over decay_cycles."""
        if self.decay_cycles is None or self.cycle >= self.decay_cycles:
            return self.epsilon
        t = self.cycle / self.decay_cycles
        return self.epsilon_start + t * (self.epsilon - self.epsilon_start)

    def _refresh_unseen_if_empty(self):
        """When explore pool empties, refill with queries stale for stale_cycles cycles."""
        if self.unseen or self.stale_cycles is None or not self._all_query_ids:
            return
        threshold = self.cycle - self.stale_cycles
        self.unseen = {
            qid for qid in self._all_query_ids
            if self.last_updated.get(qid, -1) < threshold and qid not in self.J_t
        }

    def _heap_pop_top(self, n):
        """Pop up to n queries by highest mean score, skipping stale or graduated entries.
        Popped queries are NOT re-pushed — they re-enter only when update() is called.
        This prevents the same top-K queries from monopolising every mining cycle."""
        results = []
        while len(results) < n and self.heap:
            neg_score, ver, qid = heapq.heappop(self.heap)
            if self.version.get(qid, 0) == ver and qid not in self.J_t:
                results.append(qid)
        return results

    def select_global(self, n_das=None, epsilon=None):
        """Return n_das query IDs: exploit top-score heap + explore unseen/stale pool."""
        self.cycle += 1
        self._refresh_unseen_if_empty()
        if n_das is None:
            n_das = self.n_das
        eps         = self._current_epsilon()
        n_exploit   = int(n_das * (1 - eps))
        n_explore   = n_das - n_exploit
        exploit_ids = self._heap_pop_top(n_exploit)
        explore_pool = list(self.unseen - self.J_t)
        explore_ids  = random.sample(explore_pool, min(n_explore, len(explore_pool)))
        return exploit_ids + explore_ids

    def update(self, qid, score_observed):
        """Update running mean score, push fresh heap entry, check J_t graduation."""
        n = self.n_pulls.get(qid, 0) + 1
        self.n_pulls[qid] = n
        old = self.mean_score.get(qid, 0.0)
        self.mean_score[qid] = old + (score_observed - old) / n
        self.unseen.discard(qid)
        self.last_updated[qid] = self.cycle
        ver = self.version.get(qid, 0) + 1
        self.version[qid] = ver
        heapq.heappush(self.heap, (-self.mean_score[qid], ver, qid))
        # J_t graduation: permanently exclude queries with consistently low score
        if n >= self.min_pulls:
            if not self.J_t:
                self.J_t.add(qid)
            else:
                worst = max(self.J_t, key=lambda q: self.mean_score.get(q, 0.0))
                if self.mean_score[qid] <= self.mean_score.get(worst, 0.0):
                    self.J_t.add(qid)

    def select(self, batch_ids):
        """Legacy batch-level select for run_grass_ema.py. Scores by mean score (∞ for unseen)."""
        active = [qid for qid in batch_ids if qid not in self.J_t]
        if not active:
            return set()
        ranked = sorted(active, key=lambda q: self.mean_score.get(q, float('inf')), reverse=True)
        return set(ranked[:self.n_das])


class EpsilonGreedyBandit:
    """
    Minimal ε-greedy bandit with EMA-updated gap-index estimates.

    Selection (SelectGlobal): exploit top-(1-ε)·n distinct queries by mean_g
                              + uniform random ε·n from queries not in exploit.
    Update: EMA with constant step-size α — tracks current-model g under
            non-stationary reward (Sutton & Barto §2.5).

    Heap uses lazy deletion: every update() pushes a fresh (priority, version, qid)
    entry; stale entries are skipped on pop via the version counter. No J_t, no
    cooldown, no stale_cycles, no annealing. See plan §4.5 (CaseBandit algorithm).

    Used by the new sequential-bandit pipeline (scripts/run_grass_seq_bandit.py).
    The legacy CaseBandit class above is retained for async-GRASS and EMA.
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
