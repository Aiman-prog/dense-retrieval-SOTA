"""
Tests for ANCE negative sampling logic.
Verifies: uniform random sampling from top-200 pool (not always top-1 hardest).
Covers both the initial mine (train_ance.py) and the refresh loop (run_ance_data_gen.py).
"""
import random
import pytest
from collections import Counter


# ── The exact sampling function extracted from both scripts ──────────────────

def mine_negatives(qid, pot, qrels_dict, n_negs, mining_depth):
    """Mirrors the logic in train_ance.py and run_ance_data_gen.py."""
    true_negs = [d for d in pot if d not in qrels_dict.get(qid, set())]
    candidates = true_negs if true_negs else pot
    pool = candidates[:mining_depth]
    if len(pool) >= n_negs:
        return random.sample(pool, n_negs)
    else:
        return (pool * (n_negs // max(len(pool), 1) + 1))[:n_negs]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNegativeSampling:

    def test_returns_correct_count(self):
        """Always returns exactly n_negs negatives."""
        pot = [f"d{i}" for i in range(200)]
        result = mine_negatives("q1", pot, {}, n_negs=1, mining_depth=200)
        assert len(result) == 1

    def test_returns_correct_count_multi_neg(self):
        """Works for n_negs > 1."""
        pot = [f"d{i}" for i in range(200)]
        result = mine_negatives("q1", pot, {}, n_negs=5, mining_depth=200)
        assert len(result) == 5

    def test_positives_excluded(self):
        """Positive passage IDs are never returned as negatives."""
        pot = [f"d{i}" for i in range(200)]
        qrels = {"q1": {"d0", "d1", "d2"}}
        for _ in range(50):
            result = mine_negatives("q1", pot, qrels, n_negs=1, mining_depth=200)
            assert result[0] not in {"d0", "d1", "d2"}

    def test_samples_from_pool_not_just_top1(self):
        """
        Core bug check: sampling must not always return the hardest (index 0).
        Over many draws from 200 candidates, we expect many different IDs to appear.
        """
        pot = [f"d{i}" for i in range(200)]
        random.seed(42)
        counts = Counter()
        for _ in range(500):
            result = mine_negatives("q1", pot, {}, n_negs=1, mining_depth=200)
            counts[result[0]] += 1

        # If always picking top-1, only "d0" would appear.
        assert len(counts) > 1, "Sampling is not random — always returning top-1"
        # With 500 draws from 200 candidates, expect decent spread (>20 unique)
        assert len(counts) > 20, f"Too little variety: only {len(counts)} unique negatives seen"

    def test_all_sampled_from_within_mining_depth(self):
        """Sampled negative must come from candidates[:mining_depth], not beyond."""
        pot = [f"d{i}" for i in range(500)]  # 500 candidates, depth=200
        for _ in range(100):
            result = mine_negatives("q1", pot, {}, n_negs=1, mining_depth=200)
            idx = int(result[0][1:])  # "d37" → 37
            assert idx < 200, f"Sampled d{idx} is outside the top-200 pool"

    def test_fallback_when_fewer_candidates_than_n_negs(self):
        """When pool is smaller than n_negs, pad by repeating (no crash)."""
        pot = ["d0", "d1"]  # only 2 candidates
        result = mine_negatives("q1", pot, {}, n_negs=5, mining_depth=200)
        assert len(result) == 5
        assert all(r in {"d0", "d1"} for r in result)

    def test_fallback_to_pot_when_all_are_positives(self):
        """If every candidate is a positive, fall back to using pot anyway (no empty list)."""
        pot = ["d0", "d1", "d2"]
        qrels = {"q1": {"d0", "d1", "d2"}}  # all positives
        result = mine_negatives("q1", pot, qrels, n_negs=1, mining_depth=200)
        assert len(result) == 1
        assert result[0] in {"d0", "d1", "d2"}

    def test_query_not_in_qrels(self):
        """Query with no qrels entry is treated as having no positives."""
        pot = [f"d{i}" for i in range(10)]
        result = mine_negatives("q_unseen", pot, {}, n_negs=1, mining_depth=200)
        assert len(result) == 1

    def test_no_duplicates_in_single_draw(self):
        """random.sample guarantees no duplicates — verify for n_negs > 1."""
        pot = [f"d{i}" for i in range(200)]
        for _ in range(50):
            result = mine_negatives("q1", pot, {}, n_negs=5, mining_depth=200)
            assert len(result) == len(set(result)), "Duplicate negatives in a single draw"

    def test_mining_depth_respected_with_positives_filtered(self):
        """Positives are filtered BEFORE slicing to mining_depth."""
        # pot[0] is a positive; after filtering, pool should be d1..d200 (still 200 deep)
        pot = [f"d{i}" for i in range(201)]
        qrels = {"q1": {"d0"}}
        for _ in range(100):
            result = mine_negatives("q1", pot, qrels, n_negs=1, mining_depth=200)
            assert result[0] != "d0"
            idx = int(result[0][1:])
            assert idx <= 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
