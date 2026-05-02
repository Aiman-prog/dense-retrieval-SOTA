import math
import numpy as np


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
