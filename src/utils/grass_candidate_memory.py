"""Active candidate memory for GRASS mining.

Stores per-query candidates that were previously useful (selected negatives,
high-g, high-sigma) so they remain reachable after the stale FAISS top-P stops
surfacing them. Unioned with FAISS hits before fresh current-model rerank.

Round semantics:
    current_round is a monotonically increasing mining-call counter. In
    run_grass.py (paper Algorithm 1) it ticks once per training minibatch.

    Validity: current_round - last_update_round <= ttl_rounds.
"""

import pickle
from pathlib import Path


class CandidateMemory:
    def __init__(self, max_per_query, ttl_rounds,
                 top_g_to_store, top_sigma_to_store):
        self.max_per_query      = max_per_query
        self.ttl_rounds         = ttl_rounds
        self.top_g_to_store     = top_g_to_store
        self.top_sigma_to_store = top_sigma_to_store
        # qid -> {candidate_ids, last_update_round,
        #         last_selected_negative, last_g_selected}
        self._state = {}

    @classmethod
    def load(cls, path, **kwargs):
        path = Path(path)
        if not path.exists():
            return cls(**kwargs)
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path}: not a CandidateMemory pickle")
        return obj

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def get(self, qid, current_round):
        """Returns (candidate_ids, memory_expired).

        memory_expired is True iff an entry exists for qid but its
        last_update_round is outside the TTL window.
        """
        entry = self._state.get(qid)
        if entry is None:
            return [], False
        if current_round - entry['last_update_round'] > self.ttl_rounds:
            return [], True
        return list(entry['candidate_ids']), False

    def update(self, qid, current_round, selected_negs,
               top_g_docids=None, top_sigma_docids=None,
               top_g_value=None):
        """Merge inputs into qid's memory, preserving insertion order.

        Order: fresh evidence first (selected_negs -> top_g_docids ->
        top_sigma_docids), then existing memory. Dedupe, then cap at
        max_per_query.

        Fresh-first ordering ensures newly useful candidates are never silently
        dropped when memory is full -- they push out the oldest existing entries
        instead.
        """
        top_g_docids     = top_g_docids     or []
        top_sigma_docids = top_sigma_docids or []
        existing = self._state.get(qid, {}).get('candidate_ids', [])

        merged = []
        seen   = set()
        for docid in list(selected_negs) + list(top_g_docids) + list(top_sigma_docids) + list(existing):
            if docid is None or docid in seen:
                continue
            seen.add(docid)
            merged.append(docid)
            if len(merged) >= self.max_per_query:
                break

        self._state[qid] = {
            'candidate_ids':          merged,
            'last_update_round':      current_round,
            'last_selected_negative': selected_negs[0] if selected_negs else None,
            'last_g_selected':        top_g_value,
        }

    def has(self, qid):
        return qid in self._state

    def __len__(self):
        return len(self._state)
