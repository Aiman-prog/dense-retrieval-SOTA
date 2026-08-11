"""
Async Fast-GRASS — pilot support: stratified manifest + run validity gate.

Two jobs, both needed before a lambda sweep is worth any GPU time.

**1. A representative subset.** ``--debug`` is not one: it takes
``train_items[:512]`` after ``sorted(mix_dir.glob("*.jsonl"))``, and ``train_hq.jsonl``
sorts first, so the debug set is effectively HQ-only. A manifest names an explicit,
deterministic, source-stratified set of ``query_id``s that the preflight, the
orchestrator and the miner all consume identically.

Why the manifest is INTERLEAVED rather than concatenated: the async miner walks
manifest order in batches of ``batch_size`` and does **not** shuffle (unlike
``run_fast_grass``, which calls ``random.shuffle``). A concatenated manifest would make
whole mining batches single-source, so every cache-maintenance interval would see one
domain at a time.

``query_id`` alone is a sufficient key — ``preprocessor`` emits globally distinct
prefixes (``reasonir_hq_*``, ``reasonir_vl_*``, ``msmarco_*``). Uniqueness is asserted
at build time rather than papered over with composite keys.

**2. A validity gate.** A pilot that never consumed a refreshed mined round tells you
nothing about lambda, but looks like a successful job. ``evaluate_pilot_gate`` requires
a NUMERIC mined round (``initial_data`` does not count — it is mined from the base model
before any checkpoint exists) with ``num_refresh_total > 0``, consumed by the trainer for
at least ``min_steps`` optimizer steps, plus a live miner and a saved model.

``min_steps`` comes from the recipe's ``pilot_gate_min_steps`` (128 for the pilot, 1 for
the GPU smoke, which only has 64 steps in total). Recipes without that key — the full
``async_fast_grass`` run — have no gate at all.

Usage:
  python scripts/async_fast_grass_pilot.py build-manifest --preset pilot10 --seed 42
  python scripts/async_fast_grass_pilot.py check-gate --async_dir ... --model_dir ...
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'scripts'))

from utils.helpers import get_path  # noqa: E402


# ---- mixture layout ---------------------------------------------------------

# Fixed order: the per-source RNG is seeded from (seed, index-in-this-tuple), so a
# source's draw does not depend on which other sources were requested.
SOURCE_ORDER = ('msmarco', 'vl', 'hq')

# Exactly the filenames `preprocessor.run_setup` writes into training_mixture/
# (src/data/preprocessor.py, prepare_msmarco/vl/hq_train_data).
SOURCE_FILES = {
    'msmarco': 'train_msmarco.jsonl',
    'vl': 'train_vl.jsonl',
    'hq': 'train_hq.jsonl',
}
FILE_TO_SOURCE = {v: k for k, v in SOURCE_FILES.items()}

PRESETS = {
    # 10% of each source, exactly. 8303 + 14997 + 9700 = 33,000
    # -> steps_per_epoch = ceil(33000/64) = 516, max_steps = 1032 over 2 epochs.
    'pilot10': {'counts': {'msmarco': 8303, 'vl': 14997, 'hq': 9700}},
    # GPU wiring smoke: 1,024 records stratified over whatever the mixture holds.
    'smoke1k': {'total': 1024},
}


class ManifestError(RuntimeError):
    """The manifest cannot be built or cannot be applied to the mixture."""


def _source_of(path):
    """`train_msmarco.jsonl` -> `msmarco`, for the three files run_setup writes.

    Anything else returns ``None`` and is ignored: a stray `*.jsonl` in the mixture
    directory must not silently become a fourth stratum.
    """
    return FILE_TO_SOURCE.get(Path(path).name)


def load_mixture_with_source(mix_dir=None):
    """Read the training mixture, keeping each record's source file.

    Same ``positive_passages`` handling as ``run_fast_grass._load_train_items`` (skip
    records with no positive), so the manifest can never name an item the miner would
    then drop.

    Returns ``{source: [{query_id, query, pos_docid}, ...]}``.
    """
    mix_dir = Path(mix_dir) if mix_dir else get_path("processed") / "training_mixture"
    if not mix_dir.exists():
        raise ManifestError(f"training mixture not found at {mix_dir}")

    by_source, seen_files = {}, []
    for f_path in sorted(mix_dir.glob("*.jsonl")):
        if f_path.name.startswith('.'):
            continue
        source = _source_of(f_path)
        if source is None:
            continue
        seen_files.append(f_path.name)
        items = by_source.setdefault(source, [])
        with open(f_path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                pos = d.get('positive_passages', [])
                if not pos:
                    continue
                items.append({
                    'query_id': str(d['query_id']),
                    'query': d['query'],
                    'pos_docid': pos[0]['docid'],
                })
    by_source = {s: v for s, v in by_source.items() if v}
    if not by_source:
        found = sorted(p.name for p in mix_dir.glob("*.jsonl"))
        raise ManifestError(
            f"no usable records in {mix_dir}. Expected {sorted(SOURCE_FILES.values())} "
            f"with a non-empty 'positive_passages' field; found {found or 'nothing'}.")
    return by_source


# ---- manifest construction --------------------------------------------------

def assert_unique_query_ids(by_source):
    """Every ``query_id`` must be unique across the whole mixture.

    The manifest keys on ``query_id`` alone, and both the miner and the orchestrator
    build ``qid_to_text`` dicts, so a collision would silently drop one of the colliding
    records instead of producing a short manifest we could detect.
    """
    seen, collisions = {}, []
    for source in sorted(by_source):
        for it in by_source[source]:
            qid = it['query_id']
            if qid in seen:
                collisions.append((qid, seen[qid], source))
            else:
                seen[qid] = source
    if collisions:
        sample = collisions[:5]
        raise ManifestError(
            f"{len(collisions):,} query_id collisions across sources (e.g. {sample}). "
            f"The manifest keys on query_id alone, so colliding records would be "
            f"silently dropped by the miner's qid_to_text map.")
    return len(seen)


def stratified_counts(available, total):
    """Split ``total`` across sources proportionally, largest-remainder, exact sum.

    Plain rounding does not sum to ``total``; the largest-remainder method distributes
    the shortfall to the sources with the biggest fractional parts, so the manifest is
    exactly the requested size.
    """
    sources = [s for s in SOURCE_ORDER if s in available]
    pool = sum(available[s] for s in sources)
    if total > pool:
        raise ManifestError(
            f"requested {total:,} records but the mixture only has {pool:,}")
    exact = {s: total * available[s] / pool for s in sources}
    counts = {s: int(np.floor(exact[s])) for s in sources}
    remainder = total - sum(counts.values())
    # ties broken by SOURCE_ORDER so the split is fully deterministic
    order = sorted(sources, key=lambda s: (-(exact[s] - counts[s]), SOURCE_ORDER.index(s)))
    for s in order[:remainder]:
        counts[s] += 1
    return counts


def resolve_counts(by_source, preset=None, counts=None, total=None):
    """Turn a preset / explicit counts / a total into a per-source count dict."""
    available = {s: len(v) for s, v in by_source.items()}
    if counts:
        resolved = dict(counts)
    elif total is not None:
        resolved = stratified_counts(available, int(total))
    elif preset:
        if preset not in PRESETS:
            raise ManifestError(f"unknown preset {preset!r}; have {sorted(PRESETS)}")
        spec = PRESETS[preset]
        resolved = (dict(spec['counts']) if 'counts' in spec
                    else stratified_counts(available, spec['total']))
    else:
        raise ManifestError("one of preset / counts / total is required")

    for s, n in resolved.items():
        if s not in by_source:
            raise ManifestError(
                f"source {s!r} ({SOURCE_FILES.get(s, '?')}) requested but absent from "
                f"the mixture (found { {k: len(v) for k, v in by_source.items()} })")
        if n > available[s]:
            raise ManifestError(
                f"source {s!r}: requested {n:,} records but only {available[s]:,} "
                f"are available")
        if n < 0:
            raise ManifestError(f"source {s!r}: negative count {n}")
    return resolved


def build_manifest(by_source, counts, seed=42):
    """Deterministic stratified selection, proportionally interleaved.

    Selection: per source, sort by ``query_id`` (so the draw does not depend on file
    order), then a seeded ``choice`` without replacement, then re-sort the chosen
    indices — the manifest's within-source order is always sorted-by-``query_id``.

    Interleave: item ``i`` of a source of size ``n`` gets position key
    ``(i + 0.5) / n``, and everything is merged on that key. That spreads each source
    evenly across the manifest at its own rate, so a 64-record mining batch contains
    all three sources in roughly their global proportions.

    Returns ``[{query_id, source}, ...]``.
    """
    assert_unique_query_ids(by_source)

    picked = {}
    for source, n in counts.items():
        if n == 0:
            continue
        items = sorted(by_source[source], key=lambda it: it['query_id'])
        rng = np.random.default_rng([int(seed), SOURCE_ORDER.index(source)])
        idx = np.sort(rng.choice(len(items), size=int(n), replace=False))
        picked[source] = [items[i]['query_id'] for i in idx]

    keyed = []
    for source, qids in picked.items():
        n = len(qids)
        rank = SOURCE_ORDER.index(source)
        for i, qid in enumerate(qids):
            # (key, source rank, index) — the last two make ties deterministic
            keyed.append(((i + 0.5) / n, rank, i, source, qid))
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    return [{'query_id': qid, 'source': source} for _k, _r, _i, source, qid in keyed]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path, rows, seed, counts, preset=None):
    """Write the JSONL plus a ``.meta.json`` sidecar carrying the sha256.

    The digest is what proves every sweep arm consumed identical data — arms run in
    separate jobs, possibly days apart, and nothing else ties them together.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        'preset': preset,
        'seed': int(seed),
        'counts': {s: int(counts.get(s, 0)) for s in SOURCE_ORDER if s in counts},
        'total': len(rows),
        'sha256': _sha256(path),
        'manifest': path.name,
    }
    meta_path = path.with_suffix('.meta.json')
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def manifest_meta(path):
    """Read the sidecar if present, else synthesise one from the JSONL."""
    path = Path(path)
    meta_path = path.with_suffix('.meta.json')
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    rows = load_manifest(path)
    counts = {}
    for r in rows:
        counts[r['source']] = counts.get(r['source'], 0) + 1
    return {'preset': None, 'seed': None, 'counts': counts, 'total': len(rows),
            'sha256': _sha256(path), 'manifest': path.name}


def load_manifest(path):
    """Read a manifest JSONL into ``[{query_id, source}, ...]``, order preserved."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ManifestError(f"{path.name}:{lineno}: malformed JSON ({e})") from e
            if 'query_id' not in d:
                raise ManifestError(f"{path.name}:{lineno}: record has no 'query_id'")
            rows.append({'query_id': str(d['query_id']),
                         'source': d.get('source')})
    if not rows:
        raise ManifestError(f"manifest {path} is empty")
    seen = {r['query_id'] for r in rows}
    if len(seen) != len(rows):
        raise ManifestError(
            f"manifest {path.name} repeats {len(rows) - len(seen):,} query_id(s)")
    return rows


def apply_manifest(train_items, manifest):
    """Filter ``train_items`` to the manifest and reorder into manifest order.

    **Raises on any manifest id missing from the mixture.** Silently dropping them
    would shrink the pilot by an unrecorded amount and break comparability between
    sweep arms — the same reasoning as ``canonicalize_positives``, and just as invisible
    downstream if it were allowed to pass.
    """
    by_qid = {it['query_id']: it for it in train_items}
    out, missing = [], []
    for row in manifest:
        it = by_qid.get(row['query_id'])
        if it is None:
            missing.append(row['query_id'])
        else:
            out.append(it)
    if missing:
        raise ManifestError(
            f"{len(missing):,}/{len(manifest):,} manifest query_ids are absent from "
            f"the loaded mixture (e.g. {missing[:5]}). The manifest and the mixture "
            f"have diverged; rebuild the manifest rather than training on a subset.")
    return out


def manifest_source_counts(manifest):
    counts = {}
    for r in manifest:
        counts[r['source']] = counts.get(r['source'], 0) + 1
    return counts


def maybe_apply_manifest(train_items, manifest_path, log=print):
    """Convenience wrapper used identically by preflight, orchestrator and miner."""
    if not manifest_path:
        return train_items, None
    manifest = load_manifest(manifest_path)
    items = apply_manifest(train_items, manifest)
    meta = manifest_meta(manifest_path)
    log(f"manifest {Path(manifest_path).name}: {len(items):,} items "
        f"({manifest_source_counts(manifest)}) sha256={meta['sha256'][:12]}")
    return items, meta


# ---- run validity gate ------------------------------------------------------

_MODEL_WEIGHT_FILES = ('model.safetensors', 'pytorch_model.bin',
                       'model.safetensors.index.json', 'pytorch_model.bin.index.json')


def model_is_saved(model_dir):
    """Final weights present next to the checkpoints (config + a weights file)."""
    d = Path(model_dir)
    if not (d / 'config.json').exists():
        return False
    return any((d / f).exists() for f in _MODEL_WEIGHT_FILES)


def evaluate_pilot_gate(root, trainer_summary, model_dir, miner_failed, min_steps,
                        read_meta=None):
    """Is this pilot run valid evidence about lambda? -> ``(ok, reasons, details)``.

    Conditions, all required:

    1. a **numeric** mined round was consumed. ``initial_data`` does not count: it is
       mined from the base model before any checkpoint exists, so a run that only ever
       consumed it exercised no async loop at all;
    2. that round reports ``num_refresh_total > 0`` — otherwise the cache never
       refreshed and the run says nothing about the corrected configuration;
    3. it was active for at least ``min_steps`` optimizer steps, so the mined negatives
       actually shaped the weights;
    4. the miner did not die (``miner_failed is None``; ``supervise`` normally
       *terminates* the miner, so a zero exit code is not the expected outcome);
    5. the final model was saved.

    ``read_meta`` is injectable so the tests can drive this without a handoff tree.
    """
    if read_meta is None:
        from async_fast_grass_handoff import read_meta as _rm

        def read_meta(n):
            return _rm(root, n)

    reasons = []
    rounds = list((trainer_summary or {}).get('rounds', []))

    numeric, qualifying = [], []
    for r in rounds:
        n = int(r.get('round_no', 0))
        if n < 1:
            continue                      # initial_data is not a numeric round
        meta = read_meta(n) or {}
        refresh = int(meta.get('num_refresh_total', 0) or 0)
        steps = int(r.get('steps_active', 0) or 0)
        rec = {'round_no': n, 'steps_active': steps,
               'num_refresh_total': refresh,
               'num_replace_total': int(meta.get('num_replace_total', 0) or 0),
               'source_checkpoint_step': r.get('source_checkpoint_step')}
        numeric.append(rec)
        if refresh > 0 and steps >= int(min_steps):
            qualifying.append(rec)

    if not numeric:
        reasons.append(
            "no numeric mined round was consumed — the trainer ran entirely on "
            "initial_data, so the async loop was never exercised")
    elif not qualifying:
        best = max(numeric, key=lambda r: (r['num_refresh_total'], r['steps_active']))
        reasons.append(
            f"no numeric round met the gate: best was round {best['round_no']} with "
            f"num_refresh_total={best['num_refresh_total']} active for "
            f"{best['steps_active']} steps (need >0 refreshes and >={min_steps} steps)")

    if miner_failed is not None:
        reasons.append(
            f"miner exited with code {miner_failed}; the trainer continued on stale "
            f"mined data")
    if not model_is_saved(model_dir):
        reasons.append(f"no final model weights saved in {model_dir}")

    details = {
        'min_steps': int(min_steps),
        'rounds_consumed_numeric': numeric,
        'qualifying_rounds': qualifying,
        'miner_failed': miner_failed,
        'model_saved': model_is_saved(model_dir),
    }
    return (not reasons), reasons, details


def format_gate_report(ok, reasons, details):
    lines = ["=" * 66,
             "  ASYNC FAST-GRASS — RUN VALIDITY GATE",
             "=" * 66,
             f"  min steps on a refreshed round : {details['min_steps']}",
             f"  numeric rounds consumed        : "
             f"{len(details['rounds_consumed_numeric'])}",
             f"  qualifying rounds              : {len(details['qualifying_rounds'])}",
             f"  miner_failed                   : {details['miner_failed']}",
             f"  final model saved              : {details['model_saved']}"]
    for r in details['rounds_consumed_numeric']:
        lines.append(f"    round {r['round_no']}: active {r['steps_active']} steps | "
                     f"refresh={r['num_refresh_total']} replace={r['num_replace_total']}")
    lines.append("-" * 66)
    for why in reasons:
        lines.append(f"  ❌ {why}")
    lines.append(f"  {'PASS' if ok else 'FAIL'}  run is "
                 f"{'valid' if ok else 'INVALID'} evidence about lambda")
    lines.append("=" * 66)
    return "\n".join(lines)


# ---- CLI --------------------------------------------------------------------

def _cmd_build_manifest(args):
    by_source = load_mixture_with_source(args.mixture_dir)
    total_available = assert_unique_query_ids(by_source)
    counts = resolve_counts(by_source, preset=args.preset,
                            counts=_parse_counts(args.counts), total=args.total)
    rows = build_manifest(by_source, counts, seed=args.seed)

    name = args.name or f"{args.preset or 'manifest'}_seed{args.seed}"
    out = Path(args.out) if args.out else (
        get_path("processed") / "pilot_manifests" / f"{name}.jsonl")
    meta = write_manifest(out, rows, args.seed, counts, preset=args.preset)

    print("=" * 66)
    print("  PILOT MANIFEST")
    print("=" * 66)
    print(f"  mixture           : {len(by_source)} sources, "
          f"{total_available:,} unique query_ids")
    for s in SOURCE_ORDER:
        if s in by_source:
            print(f"    {s:<9}: {len(by_source[s]):>8,} available -> "
                  f"{counts.get(s, 0):>8,} selected")
    print(f"  total selected    : {meta['total']:,}")
    print(f"  seed              : {meta['seed']}")
    print(f"  sha256            : {meta['sha256']}")
    print(f"  written           : {out}")
    print("=" * 66)
    return 0


def _parse_counts(spec):
    if not spec:
        return None
    out = {}
    for part in spec.split(','):
        k, _, v = part.partition('=')
        out[k.strip()] = int(v)
    return out


def _cmd_check_gate(args):
    root = Path(args.async_dir)
    summary_path = Path(args.model_dir) / "async_trainer_summary.json"
    if not summary_path.exists():
        print(f"❌ trainer summary not found at {summary_path}")
        return 2
    summary = json.loads(summary_path.read_text())
    ok, reasons, details = evaluate_pilot_gate(
        root, summary, args.model_dir, args.miner_failed, args.min_steps)
    print(format_gate_report(ok, reasons, details))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build-manifest', help='deterministic stratified subset')
    b.add_argument('--preset', choices=sorted(PRESETS), default=None)
    b.add_argument('--counts', default=None,
                   help='explicit per-source counts, e.g. msmarco=100,vl=200,hq=150')
    b.add_argument('--total', type=int, default=None,
                   help='stratify this many records over the available sources')
    b.add_argument('--seed', type=int, default=42)
    b.add_argument('--name', default=None, help='basename (default <preset>_seed<N>)')
    b.add_argument('--out', default=None, help='explicit output path')
    b.add_argument('--mixture_dir', default=None)
    b.set_defaults(func=_cmd_build_manifest)

    g = sub.add_parser('check-gate', help='re-evaluate a finished run')
    g.add_argument('--async_dir', required=True)
    g.add_argument('--model_dir', required=True)
    g.add_argument('--min_steps', type=int, default=128)
    g.add_argument('--miner_failed', type=int, default=None)
    g.set_defaults(func=_cmd_check_gate)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
