import heapq
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
    cooldown, no stale_cycles, no annealing. See plan §4.5.

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
