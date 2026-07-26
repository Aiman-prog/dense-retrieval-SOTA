"""
Unit tests for the Fast-GRASS negative-cache core (NegativeCache + RetiredRegistry).
CPU-only, deterministic, synthetic data — no model download, no GPU required.

Run: python scripts/fast_grass_test.py
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.negative_cache import NegativeCache, RetiredRegistry, linear_decay

DEVICE = torch.device('cpu')


# ---- mocks (mirror GradMockModel/MockTokenizer in grass_test.py) -----------

class _MockOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class GradMockModel(nn.Module):
    """CLS = nn.Embedding(input_ids[:,0]); deterministic, real params (so a param
    change visibly changes encodings — used by the refresh test)."""
    def __init__(self, vocab=1000, hidden=8, seq_len=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.config = type('C', (), {'hidden_size': hidden})()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return _MockOutput(last_hidden_state=self.emb(input_ids))


class DropoutMockModel(nn.Module):
    """GradMockModel + a real ``nn.Dropout``, so MC passes genuinely differ.

    Used by the async cached-MCDP tests: ``dropout_only()`` must be able to flip
    this module's dropout on while everything else stays in eval, and ``T``
    encodes of the same text must produce ``T`` DIFFERENT states (proving the
    initial cache holds real stochastic samples, not a repeated deterministic
    embedding).
    """
    def __init__(self, vocab=1000, hidden=8, p=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.drop = nn.Dropout(p)
        self.config = type('C', (), {'hidden_size': hidden})()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return _MockOutput(last_hidden_state=self.drop(self.emb(input_ids)))


class _BatchEncoding(dict):
    def to(self, device):
        return _BatchEncoding({k: v.to(device) for k, v in self.items()})


class MockTokenizer:
    def __call__(self, texts, padding=True, truncation=True,
                 max_length=128, return_tensors='pt'):
        ids = torch.zeros(len(texts), 4, dtype=torch.long)
        for i, t in enumerate(texts):
            ids[i, 0] = abs(hash(t)) % 1000
        return _BatchEncoding({'input_ids': ids,
                               'attention_mask': torch.ones(len(texts), 4,
                                                            dtype=torch.long)})


# ---- fixtures --------------------------------------------------------------

def make_cfg(async_defaults=False, **over):
    cfg = dict(
        B_doc=10, m=1, selection_mode='topk', lambda_val=1.0, beta=5.0, L=1024,
        uncertainty='ema', ema_alpha=0.999,
        rho_start=0.50, rho_end=0.10, cache_update_interval=100,
        max_age_steps=8, utility_ema_decay=0.95, utility_floor=0.01,
        utility_remember_threshold=0.05, K=3, R_fraction=0.25,
        uniform_candidate_fraction=0.75, replacement_candidate_multiplier=2,
        recent_query_reservoir_size=8, reentry_top_k=5, R_size_factor=0.5,
        cache_init_seed=42, steps_per_epoch=100, total_steps=1000,
        passage_max_len=128, mc_batch_size=64)
    if async_defaults:
        # async_fast_grass_implementation_details.md, "Current Async Defaults".
        # These deliberately differ from the sequential config.yaml values
        # (lambda 1.0, rho_end 0.10, max_age_epochs 4, B_doc 100k).
        cfg.update(uncertainty='cached_mcdp', lambda_val=0.5, T=3,
                   mc_dropout_p=0.3, B_doc=32_000, rho_start=0.50, rho_end=0.25,
                   max_age_epochs=2, selection_mode='topk', m=1,
                   cache_update_interval=100, batch_size=64)
        cfg.pop('L', None)   # cached-MCDP scores all of H; L is not a knob
    cfg.update(over)
    return cfg


def make_cache(cfg=None, n_corpus=20, dim=8):
    cfg = cfg or make_cfg()
    embs = np.random.default_rng(0).standard_normal((n_corpus, dim)).astype('float32')
    c_ids = [f"d{i}" for i in range(n_corpus)]
    cache = NegativeCache.init_uniform(embs, c_ids, cfg, DEVICE, dim=dim)
    corpus_lookup = {d: f"document {d} body text" for d in c_ids}
    return cache, cfg, c_ids, embs, corpus_lookup


def _rand_unit(n, dim, seed):
    t = torch.from_numpy(np.random.default_rng(seed).standard_normal((n, dim))
                         .astype('float32'))
    return torch.nn.functional.normalize(t, dim=-1)


def make_z_mc(T, n, dim, seed=0):
    """Synthetic cached MC bank ``[T, n, dim]`` of L2-normalized states.

    Each pass is drawn independently, mimicking ``T`` genuine dropout samples.
    """
    return torch.stack([_rand_unit(n, dim, seed + t) for t in range(T)], dim=0)


def _run(name, fn):
    print(f"  {name} ...", end=' ', flush=True)
    try:
        fn()
        print("PASS")
        return True
    except AssertionError as e:
        print(f"FAIL — {e}")
        return False
    except Exception as e:
        print(f"ERROR — {type(e).__name__}: {e}")
        return False


# ---- init ------------------------------------------------------------------

def test_init_shapes_and_bijection():
    cache, cfg, c_ids, embs, _ = make_cache()
    assert cache.B_doc == 10
    assert cache.Z_student.shape == (10, 8) and cache.Z_teacher.shape == (10, 8)
    assert not cache.Z_student.requires_grad and not cache.Z_teacher.requires_grad
    # student == teacher at init
    assert torch.allclose(cache.Z_student, cache.Z_teacher)
    # bijection slot <-> docid
    assert len(set(cache.docids)) == 10
    for d, s in cache.docid_to_slot.items():
        assert cache.docids[s] == d


def test_init_clamps_and_is_seeded():
    cfg = make_cfg(B_doc=10)
    embs = np.random.default_rng(1).standard_normal((5, 8)).astype('float32')
    c_ids = [f"d{i}" for i in range(5)]
    c1 = NegativeCache.init_uniform(embs, c_ids, cfg, DEVICE, dim=8)
    c2 = NegativeCache.init_uniform(embs, c_ids, cfg, DEVICE, dim=8)
    assert c1.B_doc == 5, "B_doc must clamp to corpus size"
    assert c1.docids == c2.docids, "same seed -> same H"


# ---- score -----------------------------------------------------------------

def test_score_matches_manual():
    cache, cfg, *_ = make_cache()
    qs, qt = _rand_unit(4, 8, 7), _rand_unit(4, 8, 8)
    g, s_student, sigma = cache.score(qs, qt, lambda_val=1.0)
    exp_ss = qs @ cache.Z_student.t()
    exp_sig = (exp_ss - qt @ cache.Z_teacher.t()).abs()
    assert torch.allclose(s_student, exp_ss, atol=1e-5)
    assert torch.allclose(sigma, exp_sig, atol=1e-5)
    assert torch.allclose(g, exp_ss + exp_sig, atol=1e-5)
    assert cache.cache_score_pairs == 4 * cache.B_doc


def test_score_lambda_zero():
    cache, *_ = make_cache()
    qs, qt = _rand_unit(3, 8, 1), _rand_unit(3, 8, 2)
    g, s_student, _ = cache.score(qs, qt, lambda_val=0.0)
    assert torch.allclose(g, s_student, atol=1e-6), "lambda=0 => g == s_student"


def test_score_is_grad_free_even_with_grad_queries():
    """Selection must never build a graph through the cache, even if the caller
    passes grad-enabled query embeddings."""
    cache, *_ = make_cache()
    qs = _rand_unit(3, 8, 1).requires_grad_(True)
    qt = _rand_unit(3, 8, 2).requires_grad_(True)
    g, s_student, sigma = cache.score(qs, qt, lambda_val=1.0)
    assert not g.requires_grad and g.grad_fn is None
    assert not s_student.requires_grad and not sigma.requires_grad


# ---- cheap_scores + teacher-free MCDP cache --------------------------------

def test_cheap_scores_matches_manual_and_grad_free():
    """MCDP top-L ranking: student-only cheap scores over H, grad-free even with a
    grad-enabled query, and counted like score()."""
    cache, *_ = make_cache()
    qs = _rand_unit(4, 8, 7).requires_grad_(True)
    s = cache.cheap_scores(qs)
    assert torch.allclose(s, qs @ cache.Z_student.t(), atol=1e-5)
    assert not s.requires_grad and s.grad_fn is None
    assert cache.cache_score_pairs == 4 * cache.B_doc


def test_mcdp_cache_is_teacher_free():
    """uncertainty='mcdp' → no Z_teacher; memory_bytes counts student only; score()
    raises (EMA-only), cheap_scores() works."""
    cfg = make_cfg(uncertainty='mcdp')
    cache, *_ = make_cache(cfg=cfg)
    assert cache.Z_teacher is None
    # student-only footprint == exactly the student tensor bytes
    assert cache.memory_bytes() == (cache.Z_student.element_size() *
                                    cache.Z_student.nelement())
    qs = _rand_unit(3, 8, 1)
    assert cache.cheap_scores(qs).shape == (3, cache.B_doc)
    try:
        cache.score(qs, qs, 1.0)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "score() must raise on a teacher-free (MCDP) cache"


def test_teacher_free_maintain_preserves_invariants():
    """Maintenance on an MCDP (teacher=None) cache runs student-only refresh/replace,
    keeps the slot↔docid bijection + B_doc, and never allocates Z_teacher."""
    cfg = make_cfg(uncertainty='mcdp', B_doc=8)
    cache, cfg, c_ids, embs, corpus = make_cache(cfg=cfg, n_corpus=20)
    model, tok = GradMockModel(hidden=8).eval(), MockTokenizer()
    # force everything replace-eligible
    cache.intervals_since_selected[:] = cfg['K']
    cache.utility_ema[:] = 0.0
    qs = _rand_unit(4, 8, 3)
    reservoir = {'q_student': qs, 'q_teacher': None,
                 'qids': [f"q{i}" for i in range(4)]}
    counters = cache.maintain(model, None, tok, corpus, c_ids, reservoir,
                              step=50, cfg=cfg, device=DEVICE, qrels_dict={})
    assert cache.Z_teacher is None
    assert len(cache.docids) == cache.B_doc == 8
    assert len(set(cache.docids)) == cache.B_doc
    for d, s in cache.docid_to_slot.items():
        assert cache.docids[s] == d
    assert 'num_replace' in counters and 'cache_turnover_rate' in counters


# ---- mask + select ---------------------------------------------------------

def test_mask_excludes_positive_from_selection():
    cache, *_ = make_cache()
    g = torch.zeros(1, cache.B_doc)
    g[0, 3] = 10.0   # would be the obvious top-1
    g[0, 7] = 5.0    # runner-up
    qrels = {'q': {cache.docids[3]}}
    masked = cache.mask_positives(g, ['q'], qrels)
    assert masked[0, 3] == float('-inf')
    assert g[0, 3] == 10.0, "must not mutate caller's tensor by default"
    slots, _ = cache.select(masked, m=1, mode='topk')
    assert slots[0, 0].item() == 7, "masked positive must not be selected"


def test_select_topk_picks_max():
    cache, *_ = make_cache()
    g = torch.tensor([[0.1, 0.9, 0.3, 0.7, 0.2, 0.0, 0.5, 0.4, 0.6, 0.8]])
    slots, docids = cache.select(g, m=2, mode='topk')
    assert slots[0].tolist() == [1, 9], "top-2 by g"
    assert docids[0] == [cache.docids[1], cache.docids[9]]


def test_select_softmax_highbeta_approx_topk():
    cache, *_ = make_cache()
    g = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 5.0]])
    slots, _ = cache.select(g, m=1, mode='softmax', beta=50.0)
    assert slots[0, 0].item() == 9, "huge beta -> Gumbel-softmax ~ argmax"


def test_select_softmax_distinct_and_masks():
    cache, *_ = make_cache()
    g = torch.arange(cache.B_doc, dtype=torch.float32).unsqueeze(0).clone()
    g[0, 9] = float('-inf')   # masked top
    slots, _ = cache.select(g, m=3, mode='softmax', beta=5.0)
    picked = slots[0].tolist()
    assert len(set(picked)) == 3, "Gumbel-top-k returns m distinct slots"
    assert 9 not in picked, "masked slot can never be selected"


def test_select_softmax_is_seeded_reproducible():
    """Gumbel-Softmax uses the cache's seeded RNG, not the global torch RNG."""
    cache, *_ = make_cache()
    g = torch.randn(4, cache.B_doc)
    cache._gen.manual_seed(123)
    a, _ = cache.select(g, m=2, mode='softmax', beta=5.0)
    cache._gen.manual_seed(123)
    b, _ = cache.select(g, m=2, mode='softmax', beta=5.0)
    assert torch.equal(a, b), "same seed -> identical Softmax selection"


def test_select_softmax_requires_L_ge_m():
    cache, *_ = make_cache()
    g = torch.randn(1, cache.B_doc)
    try:
        cache.select(g, m=3, mode='softmax', beta=5.0, L=2)
        raised = False
    except ValueError:
        raised = True
    assert raised, "softmax with L < m must raise (prefilter leaves < m candidates)"


def test_select_raises_when_fewer_than_m_finite():
    cache, *_ = make_cache()
    g = torch.full((1, cache.B_doc), float('-inf'))
    g[0, 0] = 1.0   # only ONE finite slot
    try:
        cache.select(g, m=2, mode='topk')
        raised = False
    except ValueError:
        raised = True
    assert raised, "must refuse to pick m negatives from < m finite slots"


# ---- utility update --------------------------------------------------------

def test_utility_update_formula_and_reset():
    cache, cfg, *_ = make_cache()
    cache.selected_indicator[torch.tensor([0, 1])] = True
    cache._update_utility(cfg)
    exp = 1.0 - cfg['utility_ema_decay']   # 0.05
    assert abs(cache.utility_ema[0].item() - exp) < 1e-6
    assert abs(cache.utility_ema[5].item() - 0.0) < 1e-6
    assert abs(cache.peak_utility_ema[1].item() - exp) < 1e-6
    assert cache.lifetime_selected_count[0].item() == 1
    assert cache.intervals_since_selected[0].item() == 0
    assert cache.intervals_since_selected[5].item() == 1
    assert not cache.selected_indicator.any(), "selected_indicator reset"


# ---- budget ----------------------------------------------------------------

def test_linear_decay_endpoints():
    assert abs(linear_decay(0.5, 0.1, 0.0) - 0.5) < 1e-9
    assert abs(linear_decay(0.5, 0.1, 1.0) - 0.1) < 1e-9
    assert abs(linear_decay(0.5, 0.1, -5) - 0.5) < 1e-9   # clamps
    assert abs(linear_decay(0.5, 0.1, 0.5) - 0.3) < 1e-9


def test_interval_budget_value():
    cache, _, *_ = make_cache()
    cfg = make_cfg(steps_per_epoch=100, total_steps=1000)
    assert cache._interval_budget(0, cfg) == 5      # rho .5 * 10 * 100/100
    assert cache._interval_budget(1000, cfg) == 1   # rho .1 * 10


def test_plan_actions_respects_budget():
    cache, cfg, *_ = make_cache()
    # make every slot over-age and eligible for action
    cache.last_refreshed_step[:] = 0
    cache.utility_ema[:5] = 0.5                    # useful -> refresh
    cache.intervals_since_selected[5:] = cfg['K']  # low-utility -> replace
    cache.utility_ema[5:] = 0.0
    refresh, replace, diag = cache._plan_actions(step=100, cfg=cfg, budget=2)
    assert len(refresh) + len(replace) <= 2, "must not exceed budget"
    assert diag['num_over_age'] == 10
    assert diag['over_age_backlog'] == 10 - 2


# ---- refresh ---------------------------------------------------------------

def test_refresh_updates_state_and_age():
    cache, cfg, *_ , corpus = make_cache()
    student = GradMockModel().eval()
    tok = MockTokenizer()
    slots = torch.tensor([0, 1])
    before = cache.Z_student[slots].clone()
    cache._refresh(slots, student, student, tok, corpus, step=42, cfg=cfg,
                   device=DEVICE)
    assert not torch.allclose(cache.Z_student[slots], before), "Z refreshed"
    assert (cache.last_refreshed_step[slots] == 42).all(), "age reset"
    # refreshed states are L2-normalized encodings
    assert torch.allclose(cache.Z_student[0].norm(), torch.tensor(1.0), atol=1e-4)


# ---- maintain integration: invariant + safety ------------------------------

def test_grace_protects_new_and_init_docs():
    """Freshly inserted docs (utility=0) and the whole cache at init must not be
    flagged low-utility until they survive K maintenance intervals (issue #1)."""
    cache, cfg, *_ = make_cache()      # init: utility_ema=0, last_refreshed=0
    grace = cfg['K'] * cfg['cache_update_interval']
    # early step (age < grace): nothing replace-eligible despite utility 0
    _, replace_early, _ = cache._plan_actions(step=grace - 1, cfg=cfg,
                                              budget=cache.B_doc)
    assert len(replace_early) == 0, "init/young docs must be grace-protected"
    # make slot 3 old, the rest freshly resident; only slot 3 may be evicted
    cache.last_refreshed_step[:] = grace + 100
    cache.last_refreshed_step[3] = 0
    step = grace + 100
    _, replace_late, _ = cache._plan_actions(step=step, cfg=cfg, budget=cache.B_doc)
    assert 3 in replace_late.tolist(), "doc past grace with util=0 is eligible"
    assert all(s not in replace_late.tolist()
               for s in range(cache.B_doc) if s != 3), "young docs still protected"


def test_maintain_preserves_invariants_and_restores_both_models():
    cache, cfg, c_ids, embs, corpus = make_cache()
    student = GradMockModel().train()      # training mode on entry
    teacher = GradMockModel().train()      # external model — must be restored too
    tok = MockTokenizer()
    # force replacements (intervals_since_selected >= K branch, grace-independent)
    cache.intervals_since_selected[:] = cfg['K'] + 1
    cache.utility_ema[:] = 0.0
    reservoir = {'q_student': _rand_unit(4, 8, 3),
                 'q_teacher': _rand_unit(4, 8, 4),
                 'qids': [f"q{i}" for i in range(4)]}
    before_docids = list(cache.docids)
    counters = cache.maintain(student, teacher, tok, corpus, c_ids, reservoir,
                              step=50, cfg=cfg, device=DEVICE, qrels_dict={})
    assert len(cache.docids) == cache.B_doc, "H size invariant (B_doc constant)"
    assert len(cache.docid_to_slot) == cache.B_doc
    assert student.training, "maintain must restore student.training"
    assert teacher.training, "maintain must restore teacher.training (issue #2)"
    assert counters['num_replace'] >= 1
    assert cache.docids != before_docids, "some slot was replaced"
    assert not cache.Z_student.requires_grad and cache.Z_student.grad is None
    # replacement adds no encoder calls beyond recertification (issue #9)
    assert counters['doc_encoder_calls_cache_replace'] == 0
    assert (counters['doc_encoder_calls_recertify'] ==
            counters['num_recertified_candidates'])


def test_replace_keeps_uniform_candidates_dominant():
    """Even with a richly populated R, uniform candidates stay >= the configured
    fraction and R is capped accordingly (issue #8). Uses uniform_candidate_fraction
    DRIFTED away from (1 - R_fraction) so the old R_fraction-only logic would fail."""
    cfg = make_cfg(uniform_candidate_fraction=0.9, replacement_candidate_multiplier=2)
    cache, _, c_ids, embs, corpus = make_cache(cfg=cfg, n_corpus=60)
    student = GradMockModel().eval()
    tok = MockTokenizer()
    # fill R with many eligible (non-H) docs so it could dominate if unbounded
    for d in [x for x in c_ids if x not in set(cache.docids)]:
        cache.registry.admit(d, {'lifetime_selected_count': 5,
                                 'peak_utility_ema': 1.0}, 0)
    cache.intervals_since_selected[:] = cfg['K'] + 1
    cache.utility_ema[:] = 0.0
    reservoir = {'q_student': _rand_unit(4, 8, 5),
                 'q_teacher': _rand_unit(4, 8, 6),
                 'qids': [f"q{i}" for i in range(4)]}
    c = cache.maintain(student, student, tok, corpus, c_ids, reservoir,
                       step=50, cfg=cfg, device=DEVICE, qrels_dict={})
    num_cand = c['num_recertified_candidates']
    assert num_cand > 0 and c['num_R_candidates'] > 0, "R should be exercised"
    min_uniform = int(np.ceil(cfg['uniform_candidate_fraction'] * num_cand))
    assert c['num_uniform_candidates'] >= min_uniform, (
        f"uniform {c['num_uniform_candidates']} < required {min_uniform}")
    assert c['num_R_candidates'] <= num_cand - min_uniform + 1


# ---- registry R ------------------------------------------------------------

def test_registry_admission_rule():
    r = RetiredRegistry(max_size=10, utility_remember_threshold=0.05)
    assert r.admit('a', {'lifetime_selected_count': 1, 'peak_utility_ema': 0.0}, 1)
    assert r.admit('b', {'lifetime_selected_count': 0, 'peak_utility_ema': 0.1}, 1)
    assert not r.admit('c', {'lifetime_selected_count': 0, 'peak_utility_ema': 0.01}, 1)
    assert 'c' not in r.entries and len(r) == 2


def test_registry_bound_keeps_strongest():
    r = RetiredRegistry(max_size=2, utility_remember_threshold=0.05)
    for i, pk in enumerate([0.1, 0.9, 0.5, 0.8, 0.2]):
        r.admit(f"d{i}", {'lifetime_selected_count': 1, 'peak_utility_ema': pk}, i)
    assert len(r) == 2
    kept = set(r.entries.keys())
    assert kept == {'d1', 'd3'}, f"strongest by peak utility, got {kept}"


def test_registry_nominate_subset():
    r = RetiredRegistry(max_size=10, utility_remember_threshold=0.05)
    for i in range(5):
        r.admit(f"d{i}", {'lifetime_selected_count': 1, 'peak_utility_ema': 0.5}, i)
    nom = r.nominate(3, np.random.default_rng(0))
    assert len(nom) == 3 and set(nom) <= set(r.entries.keys())


# ---- recertification (controlled R-only candidate set) ---------------------

def test_recertification_inserts_top_reentry_candidate():
    """Force uniform_candidate_fraction=0 so candidates == registry nominations
    (controlled), then verify the inserted doc is the one with the highest
    re-entry score, and that it is removed from R once reinserted (issue #7)."""
    cfg = make_cfg(B_doc=4, replacement_candidate_multiplier=2,
                   uniform_candidate_fraction=0.0, reentry_top_k=1, max_age_steps=8)
    embs = np.random.default_rng(0).standard_normal((12, 8)).astype('float32')
    c_ids = [f"d{i}" for i in range(12)]
    cache = NegativeCache.init_uniform(embs, c_ids, cfg, DEVICE, dim=8)
    corpus = {d: f"text for {d}" for d in c_ids}
    student = GradMockModel().eval()
    tok = MockTokenizer()

    # candidate docids not currently in H
    cand_ids = [d for d in c_ids if d not in set(cache.docids)][:2]
    reg = RetiredRegistry(max_size=10, utility_remember_threshold=0.0)
    for d in cand_ids:
        reg.admit(d, {'lifetime_selected_count': 1, 'peak_utility_ema': 1.0}, 0)
    cache.registry = reg

    # reservoir query = encoding of the FIRST candidate's text -> it should win
    from utils.helpers import encode_batch_tensor
    q = encode_batch_tensor(student, tok, [corpus[cand_ids[0]]], DEVICE,
                            128, 64, requires_grad=False)
    reservoir = {'q_student': q, 'q_teacher': q, 'qids': ['q0']}

    slots = torch.tensor([0])            # replace slot 0
    cache._replace(slots, student, student, tok, corpus, c_ids, reservoir,
                   step=10, cfg=cfg, device=DEVICE, qrels_dict={})
    assert cache.docids[0] == cand_ids[0], (
        f"highest-reentry candidate should be inserted, got {cache.docids[0]}")
    assert cand_ids[0] not in cache.registry.entries, (
        "reinserted doc must leave R (retired-only registry)")
    assert len(cache.docids) == cache.B_doc


def test_recertification_skips_all_masked_candidates():
    """If every candidate is a known positive of every reservoir query (reentry
    = -inf), none may be promoted into H (issue #6)."""
    cfg = make_cfg(B_doc=4, replacement_candidate_multiplier=2,
                   uniform_candidate_fraction=0.0, reentry_top_k=1, max_age_steps=8)
    embs = np.random.default_rng(0).standard_normal((12, 8)).astype('float32')
    c_ids = [f"d{i}" for i in range(12)]
    cache = NegativeCache.init_uniform(embs, c_ids, cfg, DEVICE, dim=8)
    corpus = {d: f"text for {d}" for d in c_ids}
    student = GradMockModel().eval()
    tok = MockTokenizer()
    cand_ids = [d for d in c_ids if d not in set(cache.docids)][:2]
    reg = RetiredRegistry(max_size=10, utility_remember_threshold=0.0)
    for d in cand_ids:
        reg.admit(d, {'lifetime_selected_count': 1, 'peak_utility_ema': 1.0}, 0)
    cache.registry = reg
    from utils.helpers import encode_batch_tensor
    q = encode_batch_tensor(student, tok, ["anything"], DEVICE, 128, 64,
                            requires_grad=False)
    reservoir = {'q_student': q, 'q_teacher': q, 'qids': ['q0']}
    # mark BOTH candidates as positives of q0 -> all g masked -> reentry -inf
    qrels = {'q0': set(cand_ids)}
    before = list(cache.docids)
    rc = cache._replace(torch.tensor([0]), student, student, tok, corpus, c_ids,
                        reservoir, step=10, cfg=cfg, device=DEVICE, qrels_dict=qrels)
    assert rc['num_replace'] == 0, "no finite-reentry candidate -> no insertion"
    assert cache.docids == before, "H unchanged when all candidates masked"


# ---- runner ----------------------------------------------------------------

TESTS = [
    ("init: shapes + slot/docid bijection + no-grad Z", test_init_shapes_and_bijection),
    ("init: clamps B_doc to corpus + seeded", test_init_clamps_and_is_seeded),
    ("score: g = s_hat + lambda*|s_stu - s_tea|", test_score_matches_manual),
    ("score: lambda=0 => g == s_student", test_score_lambda_zero),
    ("score: grad-free even with grad queries (#3)", test_score_is_grad_free_even_with_grad_queries),
    ("cheap_scores: student-only, grad-free, counted", test_cheap_scores_matches_manual_and_grad_free),
    ("mcdp cache: teacher-free (Z_teacher None, score() raises)", test_mcdp_cache_is_teacher_free),
    ("mcdp maintain: teacher-free invariants preserved", test_teacher_free_maintain_preserves_invariants),
    ("mask: positive excluded from selection", test_mask_excludes_positive_from_selection),
    ("select: TopK picks max-g", test_select_topk_picks_max),
    ("select: Softmax high-beta ~ TopK", test_select_softmax_highbeta_approx_topk),
    ("select: Gumbel-top-k distinct + masks", test_select_softmax_distinct_and_masks),
    ("select: Softmax seeded/reproducible (#4)", test_select_softmax_is_seeded_reproducible),
    ("select: Softmax requires L >= m", test_select_softmax_requires_L_ge_m),
    ("select: raises when < m finite slots (#5)", test_select_raises_when_fewer_than_m_finite),
    ("utility: EMA formula + indicator reset", test_utility_update_formula_and_reset),
    ("budget: linear_decay endpoints", test_linear_decay_endpoints),
    ("budget: interval budget value", test_interval_budget_value),
    ("budget: plan_actions respects budget", test_plan_actions_respects_budget),
    ("grace: protects new/init docs from churn (#1)", test_grace_protects_new_and_init_docs),
    ("refresh: state updated + age reset", test_refresh_updates_state_and_age),
    ("maintain: invariants + restores both models (#2,#9)", test_maintain_preserves_invariants_and_restores_both_models),
    ("replace: uniform candidates stay dominant (#8)", test_replace_keeps_uniform_candidates_dominant),
    ("registry: admission rule", test_registry_admission_rule),
    ("registry: bound keeps strongest", test_registry_bound_keeps_strongest),
    ("registry: nominate subset of R", test_registry_nominate_subset),
    ("recertify: top-reentry inserted + leaves R (#6,#7)", test_recertification_inserts_top_reentry_candidate),
    ("recertify: all-masked candidates skipped (#6)", test_recertification_skips_all_masked_candidates),
]


def main():
    print("\nFast-GRASS negative-cache unit tests")
    print("=" * 55)
    passed = sum(_run(name, fn) for name, fn in TESTS)
    total = len(TESTS)
    print("=" * 55)
    print(f"  {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
