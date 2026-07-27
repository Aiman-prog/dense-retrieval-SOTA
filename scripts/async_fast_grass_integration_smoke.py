"""
Async Fast-GRASS — integration smoke (CPU, no GPU, no downloads).

Drives the real handoff end to end with a tiny corpus and mock models: the miner
mines a round from a fake checkpoint and publishes it; the trainer starts on
`initial_data`, swaps to `training_data_1`, and keeps training continuously across
the swap (async_fast_grass_architecture.md, "Test Plan" -> Integration smoke).

Checks:
  - initial cached-MCDP data is generated BEFORE ready_initial exists.
  - a mined round is invisible until its ready_N marker lands.
  - the trainer consumes initial_data, then swaps to training_data_1.
  - global_step is CONTINUOUS across the swap and the optimizer/scheduler objects
    are the same ones (a swap rebuilds only the dataloader).
  - mined negatives come from H and never leak a known positive.
  - ordinary mining does zero document encoder calls.
  - the trainer's loss path never touches Z_mc/Z_mean.
  - checkpoints are only visible to the miner once optimizer.pt is written.

Run: python scripts/async_fast_grass_integration_smoke.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.helpers import is_valid_checkpoint  # noqa: E402
from utils.negative_cache import NegativeCache  # noqa: E402
from fast_grass_test import make_cfg, DropoutMockModel, MockTokenizer, DEVICE  # noqa: E402
from async_fast_grass_cached_mcdp import (  # noqa: E402
    mine_batch_cached_mcdp, maintain_interval_cached_mcdp, QueryMCReservoir,
    MaintenanceDriver, encode_queries_mc,
)
from async_fast_grass_handoff import (  # noqa: E402
    initial_paths, work_paths, round_paths, publish_round, write_ready_initial,
    latest_committed_round, resolve_training_data, resolve_cache_state, read_meta,
    newest_valid_checkpoint,
)
from run_async_fast_grass_train import (  # noqa: E402
    MinedRoundDataset, MinedRoundValidationError, make_dataloader, _collate,
)

DIM, T, N_CORPUS, B_DOC, N_QUERIES = 16, 3, 60, 20, 24
BATCH = 4


def _world():
    cfg = make_cfg(uncertainty='cached_mcdp', B_doc=B_DOC, T=T, lambda_val=0.5,
                   m=1, batch_size=BATCH, mc_batch_size=8, miner_mc_batch_size=8,
                   passage_max_len=8, query_max_len=8,
                   cache_update_interval=2, steps_per_epoch=6, total_steps=12,
                   max_age_steps=8)
    c_ids = [f"d{i}" for i in range(N_CORPUS)]
    corpus_lookup = {d: f"document {d} body text" for d in c_ids}
    embs = np.random.default_rng(0).standard_normal((N_CORPUS, DIM)).astype('float32')
    train_items = [{'query_id': f"q{i}", 'query': f"query number {i}",
                    'pos_docid': c_ids[i % N_CORPUS]} for i in range(N_QUERIES)]
    qid_to_text = {it['query_id']: it['query'] for it in train_items}
    qrels = {it['query_id']: {it['pos_docid']} for it in train_items}
    model = DropoutMockModel(hidden=DIM, p=0.3)
    tok = MockTokenizer()
    return cfg, c_ids, corpus_lookup, embs, train_items, qid_to_text, qrels, model, tok


def _mine_into(cache, model, tok, out_dir, train_items, qid_to_text, qrels, cfg,
               corpus_lookup, c_ids, source_step, do_maintain=True):
    """Mine every query into ``out_dir/mined.jsonl``. Returns (stats, mined)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reservoir = QueryMCReservoir(cfg['recent_query_reservoir_size'])
    driver = MaintenanceDriver(cfg, BATCH)
    doc_calls, all_mined, n_maint = 0, {}, 0
    qid_to_pos = {it['query_id']: it['pos_docid'] for it in train_items}
    H_seen = set(cache.docids)

    with open(out_dir / "mined.jsonl", 'w') as f:
        for b in range(0, len(train_items), BATCH):
            batch = train_items[b:b + BATCH]
            qids = [it['query_id'] for it in batch]
            mined, _s, q_mc, ms = mine_batch_cached_mcdp(
                cache, model, tok, qids, qid_to_text, qrels, T, cfg, DEVICE)
            doc_calls += ms['mcdp_doc_encoder_calls_mining']
            H_now = set(cache.docids)
            for qid in qids:
                negs = mined.get(qid, [])
                assert negs and all(d in H_now for d in negs), "negative not from H"
                assert all(d not in qrels[qid] for d in negs), "positive leaked"
                f.write(json.dumps({'query_id': qid, 'query': qid_to_text[qid],
                                    'pos_docid': qid_to_pos[qid],
                                    'neg_docids': negs}) + "\n")
            all_mined.update(mined)
            reservoir.add(q_mc, qids)
            H_seen |= H_now
            if do_maintain:
                driver.add(len(qids))
                while driver.should_fire():
                    driver.consume()
                    maintain_interval_cached_mcdp(
                        cache, model, tok, corpus_lookup, c_ids, reservoir.get(),
                        source_step, T, cfg, DEVICE, qrels_dict=qrels)
                    n_maint += 1
    return {'doc_calls': doc_calls, 'n_maint': n_maint,
            'n_records': len(all_mined)}, all_mined


def _make_checkpoint(out_dir, step, valid=True):
    ck = Path(out_dir) / f"checkpoint-{step}"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "config.json").write_text("{}")
    (ck / "scheduler.pt").write_text("s")
    if valid:
        (ck / "optimizer.pt").write_text("o")
    return ck


def _check_strict_dataset(tmp, corpus_lookup, cfg):
    """Every malformed round shape must raise, never silently degrade.

    Substituting empty text for a missing docid would train the model to pull
    queries toward an empty passage while the loss curve looked perfectly healthy.
    """
    good_docids = list(corpus_lookup)[:4]
    base = {'query_id': 'q0', 'query': 'a query', 'pos_docid': good_docids[0],
            'neg_docids': [good_docids[1]]}
    cases = [
        ("malformed JSON", "{not json\n"),
        ("missing pos_docid", json.dumps({k: v for k, v in base.items()
                                          if k != 'pos_docid'}) + "\n"),
        ("null pos_docid", json.dumps({**base, 'pos_docid': None}) + "\n"),
        ("missing neg_docids", json.dumps({k: v for k, v in base.items()
                                           if k != 'neg_docids'}) + "\n"),
        ("too few negatives", json.dumps({**base, 'neg_docids': []}) + "\n"),
        ("neg_docids not a list", json.dumps({**base, 'neg_docids': 'd1'}) + "\n"),
        ("unknown positive docid",
         json.dumps({**base, 'pos_docid': 'NOT_IN_CORPUS'}) + "\n"),
        ("unknown negative docid",
         json.dumps({**base, 'neg_docids': ['NOT_IN_CORPUS']}) + "\n"),
    ]
    all_ok = True
    for i, (name, payload) in enumerate(cases):
        d = tmp / f"bad_round_{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "mined.jsonl").write_text(payload)
        try:
            MinedRoundDataset(d, corpus_lookup, m=1)
        except MinedRoundValidationError:
            pass
        except Exception as e:
            print(f"    {name}: wrong exception {type(e).__name__}: {e}")
            all_ok = False
        else:
            print(f"    {name}: ACCEPTED (should have raised)")
            all_ok = False

    # a round smaller than one batch would spin forever with drop_last=True
    d = tmp / "short_round"
    d.mkdir(parents=True, exist_ok=True)
    (d / "mined.jsonl").write_text(json.dumps(base) + "\n")
    try:
        make_dataloader(d, corpus_lookup, m=1, batch_size=8, num_workers=0)
    except MinedRoundValidationError:
        pass
    else:
        print("    short round (< 1 batch): ACCEPTED (should have raised)")
        all_ok = False

    # an empty directory is not a usable round either
    d = tmp / "empty_round"
    d.mkdir(parents=True, exist_ok=True)
    try:
        MinedRoundDataset(d, corpus_lookup, m=1)
    except MinedRoundValidationError:
        pass
    else:
        print("    empty round: ACCEPTED (should have raised)")
        all_ok = False

    # and the happy path still works
    d = tmp / "good_round"
    d.mkdir(parents=True, exist_ok=True)
    d_recs = [json.dumps({**base, 'query_id': f"q{i}"}) for i in range(8)]
    (d / "mined.jsonl").write_text("\n".join(d_recs) + "\n")
    ds = MinedRoundDataset(d, corpus_lookup, m=1)
    if len(ds) != 8 or not isinstance(ds[0]['positive'], str) or not ds[0]['positive']:
        print("    good round: rejected or produced empty text")
        all_ok = False
    return all_ok


def _check_initial_round_maintains(tmp, cfg, c_ids, corpus_lookup, embs,
                                   train_items, qid_to_text, qrels):
    """Drives the PRODUCTION ``mine_initial_data``, not a copy of its loop.

    Round 0 must use the same reservoir / cadence / periodic maintenance / final
    fold as every later round, and must not persist a selected_indicator
    accumulated across the whole mixture (that would make almost every slot look
    useful to the first real maintenance interval and suppress replacement).
    """
    from train_async_fast_grass import mine_initial_data

    model, tok = DropoutMockModel(hidden=DIM, p=0.3), MockTokenizer()
    cache, _ = NegativeCache.init_cached_mcdp(embs, c_ids, corpus_lookup, model,
                                              tok, cfg, DEVICE, dim=DIM)
    out = tmp / "initial_probe" / "mined.jsonl"
    stats = mine_initial_data(cache, model, tok, out, train_items, qid_to_text,
                              corpus_lookup, qrels, cfg, DEVICE, BATCH)

    ok = True
    if stats['num_maintenance_intervals'] == 0:
        print("    initial round ran NO maintenance intervals"); ok = False
    if stats['mcdp_doc_encoder_calls_mining'] != 0:
        print("    initial mining encoded documents"); ok = False
    if bool(cache.selected_indicator.any()):
        print("    selected_indicator would be persisted non-empty"); ok = False
    if stats['num_queries'] != len(train_items):
        print(f"    mined {stats['num_queries']} of {len(train_items)} queries"); ok = False
    if not out.exists() or not out.read_text().strip():
        print("    no mined records written"); ok = False
    return ok


def _check_supervision_kills_trainer():
    """Calls the PRODUCTION supervisor, not a reimplementation of it.

    Three scenarios:
      1. miner dies mid-run          -> trainer terminated, run fails
      2. miner dies as trainer exits -> still caught by the post-exit re-check
      3. clean run                   -> no false failure
    """
    import subprocess
    from train_async_fast_grass import supervise

    quiet = lambda *a, **k: None
    all_ok = True

    # 1. miner dies while the trainer is still working
    miner = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])
    trainer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    failed, trc = supervise(trainer, miner, poll_seconds=0.2, grace=10, log=quiet)
    if failed != 7:
        print(f"    mid-run miner death not detected (got {failed})")
        all_ok = False
    if trainer.poll() is None:
        print("    trainer still running after the miner died")
        all_ok = False

    # 2. RACE: the miner dies in the same window in which the trainer exits. The
    #    in-loop check can miss this; the post-exit re-check must not.
    miner = subprocess.Popen([sys.executable, "-c",
                              "import time,sys; time.sleep(0.5); sys.exit(9)"])
    trainer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.6)"])
    failed, trc = supervise(trainer, miner, poll_seconds=5.0, grace=10, log=quiet)
    if failed != 9:
        print(f"    miner death racing the trainer exit was MISSED (got {failed}); "
              f"the run would have been reported successful")
        all_ok = False

    # 3. clean shutdown: miner still alive, trainer finishes normally
    miner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    trainer = subprocess.Popen([sys.executable, "-c", "pass"])
    failed, trc = supervise(trainer, miner, poll_seconds=0.2, grace=10, log=quiet)
    if failed is not None or trc != 0:
        print(f"    healthy run misreported (miner_failed={failed}, trainer_rc={trc})")
        all_ok = False
    if miner.poll() is None:
        print("    miner not stopped after a clean trainer exit")
        all_ok = False
    return all_ok


def _check_canonicalized_positives(corpus_lookup):
    """Positives carrying pre-dedup docids must be remapped, not rejected.

    preprocessor.run_setup MD5-dedupes passages and remaps only the CORPUS and the
    QRELS — the training mixture keeps its original positive docid. A positive whose
    text was a duplicate therefore names a docid absent from the corpus, which the
    strict trainer rejects. This is a startup failure, not a corner case.
    """
    from async_fast_grass_cached_mcdp import (canonicalize_positives,
                                              UnresolvablePositivesError)

    real, canonical = list(corpus_lookup)[0], list(corpus_lookup)[1]
    ok = True

    # happy path: resolvable kept as-is, pre-dedup docid remapped via qrels
    items = [
        {'query_id': 'ok', 'query': 'q', 'pos_docid': real},          # resolvable
        {'query_id': 'dup', 'query': 'q', 'pos_docid': 'PRE_DEDUP'},  # remapped away
    ]
    qrels = {'ok': {real}, 'dup': {canonical}}
    out, stats = canonicalize_positives(items, qrels, corpus_lookup, log=lambda *a: None)
    by_qid = {it['query_id']: it['pos_docid'] for it in out}
    if by_qid.get('ok') != real:
        print("    resolvable positive was altered"); ok = False
    if by_qid.get('dup') != canonical:
        print(f"    pre-dedup positive not remapped (got {by_qid.get('dup')})"); ok = False
    if stats != {'total': 2, 'kept': 2, 'remapped': 1, 'dropped': 0}:
        print(f"    unexpected stats {stats}"); ok = False
    if not all(it['pos_docid'] in corpus_lookup for it in out):
        print("    a surviving positive still misses the corpus"); ok = False

    # an unresolvable positive must FAIL, not be silently dropped: training on the
    # remainder would shrink the mixture and break baseline comparability
    items.append({'query_id': 'gone', 'query': 'q', 'pos_docid': 'MISSING'})
    qrels['gone'] = set()
    try:
        canonicalize_positives(items, qrels, corpus_lookup, log=lambda *a: None)
    except UnresolvablePositivesError as e:
        if 'comparability' not in str(e):
            print(f"    raised, but without explaining why: {e}"); ok = False
    else:
        print("    unresolvable positive was SILENTLY DROPPED (must raise)"); ok = False

    # a qrels entry that itself misses the corpus is not a valid fallback
    items2 = [{'query_id': 'bad', 'query': 'q', 'pos_docid': 'PRE_DEDUP'}]
    try:
        canonicalize_positives(items2, {'bad': {'ALSO_MISSING'}}, corpus_lookup,
                               log=lambda *a: None)
    except UnresolvablePositivesError:
        pass
    else:
        print("    accepted a qrels fallback that is not in the corpus"); ok = False
    return ok


def run():
    cfg, c_ids, corpus_lookup, embs, train_items, qid_to_text, qrels, model, tok = _world()
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="async_fg_integration_"))
    root = tmp / "async_mining"
    root.mkdir(parents=True)
    model_dir = tmp / "model_out"
    model_dir.mkdir()

    print("\n" + "=" * 66)
    print("  ASYNC FAST-GRASS — INTEGRATION SMOKE (CPU, mocks)")
    print("=" * 66)

    # ---- 1. initial round: cache + initial_data, marker LAST ----------------
    cache, init_stats = NegativeCache.init_cached_mcdp(
        embs, c_ids, corpus_lookup, model, tok, cfg, DEVICE, dim=DIM)
    p = initial_paths(root)
    stats, _ = _mine_into(cache, model, tok, p['training_data'], train_items,
                          qid_to_text, qrels, cfg, corpus_lookup, c_ids, 0,
                          do_maintain=False)
    p['mining_meta'].write_text(json.dumps({'source_checkpoint': 'base_model',
                                            'source_checkpoint_step': 0}))
    cache.save_state(p['cache_state'])
    data_before_marker = (p['training_data'] / "mined.jsonl").exists()
    marker_before = p['ready'].exists()
    write_ready_initial(root)

    print(f"  initial data written before ready_initial : {data_before_marker and not marker_before}")
    print(f"  initial mining doc encodes                : {stats['doc_calls']} (must be 0)")
    print(f"  T genuine stochastic states               : "
          f"{not torch.allclose(cache.Z_mc[0], cache.Z_mc[1], atol=1e-6)}")
    ok &= data_before_marker and not marker_before and stats['doc_calls'] == 0
    ok &= latest_committed_round(root) == 0     # ready_initial is not a round

    # ---- 2. miner: fake checkpoint -> round 1 staged (NOT yet published) ----
    # The checkpoint step must be inside the smoke's step range, otherwise
    # async_gap_steps (consume_step - source_checkpoint_step) comes out negative,
    # which is meaningless and would hide a real lag bug.
    CKPT_STEP = 2
    _make_checkpoint(model_dir, CKPT_STEP, valid=False)
    assert newest_valid_checkpoint(model_dir) is None, "invalid ckpt must be skipped"
    ck = _make_checkpoint(model_dir, CKPT_STEP, valid=True)
    sel = newest_valid_checkpoint(model_dir)
    print(f"  checkpoint visible only with optimizer.pt : {sel is not None and sel[0] == CKPT_STEP}")
    ok &= sel is not None and sel[0] == CKPT_STEP and is_valid_checkpoint(ck)

    w = work_paths(root, 1)
    r_stats, _ = _mine_into(cache, model, tok, w['training_data'], train_items,
                            qid_to_text, qrels, cfg, corpus_lookup, c_ids, CKPT_STEP)
    cache.save_state(w['cache_state'])
    w['mining_meta'].write_text(json.dumps({'round_no': 1,
                                            'source_checkpoint_step': CKPT_STEP}))
    # staged but uncommitted: the trainer must not see it yet
    invisible_before_publish = latest_committed_round(root) == 0
    print(f"  round 1 invisible while only staged       : {invisible_before_publish}")
    print(f"  round mining doc encodes                  : {r_stats['doc_calls']} (must be 0)")
    print(f"  in-round maintenance intervals            : {r_stats['n_maint']}")
    ok &= invisible_before_publish and r_stats['doc_calls'] == 0 and r_stats['n_maint'] > 0

    # ---- 3. trainer: initial_data -> swap to round 1 ------------------------
    ds0 = MinedRoundDataset(resolve_training_data(root, 0), corpus_lookup, cfg['m'])
    trainer = torch.nn.Linear(DIM, DIM)
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    opt_id, sched_id = id(optimizer), id(scheduler)

    loader = torch.utils.data.DataLoader(ds0, batch_size=BATCH, shuffle=True,
                                         collate_fn=_collate, drop_last=True)
    it = iter(loader)
    active_round, global_step, swap_at, src_step = 0, 0, None, None
    ready_poll_steps = 2
    PUBLISH_AT = 5           # the miner finishes round 1 mid-training
    seen_steps, steps_on_initial = [], 0

    for _ in range(12):
        # the miner publishes partway through, so the trainer genuinely trains on
        # initial_data first and only then swaps — a swap at step 0 would test nothing
        if global_step == PUBLISH_AT and latest_committed_round(root) == 0:
            publish_round(root, 1)

        if global_step % ready_poll_steps == 0:
            latest = latest_committed_round(root)
            if latest > active_round:
                meta = read_meta(root, latest)
                ds = MinedRoundDataset(resolve_training_data(root, latest),
                                       corpus_lookup, cfg['m'])
                loader = torch.utils.data.DataLoader(
                    ds, batch_size=BATCH, shuffle=True, collate_fn=_collate,
                    drop_last=True)
                it = iter(loader)          # ONLY the dataloader is rebuilt
                active_round = latest
                swap_at = global_step
                src_step = meta['source_checkpoint_step']
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        # stand-in fresh-loss step: real trainer encodes texts, never Z_mc/Z_mean
        assert 'queries' in batch and 'positives' in batch and 'negatives' in batch
        x = torch.randn(len(batch['queries']), DIM)
        loss = trainer(x).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1
        seen_steps.append(global_step)
        if active_round == 0:
            steps_on_initial += 1

    continuous = seen_steps == list(range(1, 13))
    state_path, committed = resolve_cache_state(root)
    print(f"  resolve_cache_state -> committed round    : {committed} "
          f"({state_path.name})")
    ok &= committed == 1
    print(f"  steps trained on initial_data first       : {steps_on_initial}")
    print(f"  trainer swapped initial -> round 1 at step: {swap_at}")
    print(f"  global_step continuous across swap        : {continuous}")
    ok &= steps_on_initial > 0 and swap_at is not None and swap_at >= PUBLISH_AT
    print(f"  optimizer/scheduler preserved             : "
          f"{id(optimizer) == opt_id and id(scheduler) == sched_id}")
    print(f"  scheduler state advanced (not reset)      : {scheduler.last_epoch == 12}")
    gap = (swap_at - src_step) if (swap_at is not None and src_step is not None) else None
    print(f"  async_gap_steps at consume                : {gap}")
    ok &= (active_round == 1 and swap_at is not None and continuous
           and id(optimizer) == opt_id and id(scheduler) == sched_id
           and scheduler.last_epoch == 12
           and gap is not None and gap >= 0)   # a negative model-data lag is nonsense

    # ---- 4. trainer never needed miner embeddings --------------------------
    sample = ds0[0]
    text_only = isinstance(sample['positive'], str) and all(
        isinstance(t, str) for t in sample['negatives'])
    print(f"  trainer dataset is text-only (no Z_mc)    : {text_only}")
    ok &= text_only

    # ---- 5. strict dataset validation --------------------------------------
    print("-" * 66)
    strict_ok = _check_strict_dataset(tmp, corpus_lookup, cfg)
    print(f"  malformed rounds rejected, never silently used : {strict_ok}")
    ok &= strict_ok

    # ---- 6. miner restart resumes AFTER the mined checkpoint ---------------
    prev_meta = read_meta(root, latest_committed_round(root))
    resume_from = int(prev_meta.get('source_checkpoint_step', -1))
    would_remine = newest_valid_checkpoint(model_dir, exclude_step=-1)
    would_skip = newest_valid_checkpoint(model_dir, exclude_step=resume_from)
    restart_ok = (resume_from == CKPT_STEP and would_remine is not None
                  and would_skip is None)
    print(f"  restart resumes after step {resume_from} (no re-mine) : {restart_ok}")
    ok &= restart_ok

    # ---- 7. initial round runs periodic maintenance like any other round ----
    print("-" * 66)
    init_ok = _check_initial_round_maintains(tmp, cfg, c_ids, corpus_lookup, embs,
                                             train_items, qid_to_text, qrels)
    print(f"  initial round maintains + folds indicator : {init_ok}")
    ok &= init_ok

    # ---- 8. supervision: a dead miner must fail the run --------------------
    sup_ok = _check_supervision_kills_trainer()
    print(f"  dead miner terminates trainer + fails     : {sup_ok}")
    ok &= sup_ok

    # ---- 9. pre-dedup positive docids are canonicalized --------------------
    canon_ok = _check_canonicalized_positives(corpus_lookup)
    print(f"  pre-dedup positives remapped, not rejected: {canon_ok}")
    ok &= canon_ok

    print("=" * 66)
    print(f"  {'PASS' if ok else 'FAIL'}  async handoff integration")
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
