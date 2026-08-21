"""
Async Fast-GRASS — lambda pilot promotion decision.

Reads the per-domain BRIGHT results that ``src/evaluation/evaluate.py`` wrote for the
lambda=0 baseline and each nonzero candidate, and applies the agreed rule:

    promote      macro NDCG@10 delta >= 0.005 AND the candidate wins >= 3 of 4 domains
    inconclusive delta in [0.002, 0.005)  -> needs a second pilot seed, not a full run
    stop         delta < 0.002, negative, or driven by a single domain
    tie-break    if both candidates promote and differ by < 0.002, take the SMALLER
                 lambda

**This is a permissive screening gate, not a statistical test.** Four BRIGHT development
domains is roughly 520 queries; the macro standard error is on the order of 0.01-0.015,
so a 0.005 threshold sits inside the noise, and "wins 3 of 4 domains" has p~0.31 under
the null. It is deliberately generous — it exists to stop clearly-useless lambdas from
consuming 16-hour jobs, not to establish that uncertainty helps. Only the matched full
confirmation (lambda=0 vs the promoted lambda, same data, same seed, corrected
max_age_steps) can support that claim. Per-domain deltas and query counts are printed
alongside the verdict so the margin is always visible next to the sample it rests on.

Refuses to run on partial results: a missing domain file is an error, never a smaller
macro average.

Usage:
  python scripts/dev/lambda_pilot_decide.py \\
    --baseline async_fast_grass_pilot_bge_m3_pilot_lam0 \\
    --candidates async_fast_grass_pilot_bge_m3_pilot_lamLOW \\
                 async_fast_grass_pilot_bge_m3_pilot_lamMED \\
    --domains biology,economics,stackoverflow,theoremqa_questions
"""
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils.helpers import load_config, get_data_base_dir, get_path  # noqa: E402

PROMOTE_DELTA = 0.005
INCONCLUSIVE_DELTA = 0.002
MIN_DOMAIN_WINS = 3
TIE_DELTA = 0.002
# NDCG values arrive as floats, so a delta that is exactly 0.005 in decimal arithmetic
# lands at 0.004999999999999977 and would be demoted to "inconclusive" by an exact
# `>=`. The thresholds are round numbers chosen by hand; representation error must not
# decide which side of one a run falls on.
EPS = 1e-9

SCREENING_CAVEAT = (
    "Permissive screening gate, not a statistical test: ~520 queries over 4 domains "
    "gives a macro standard error near 0.01-0.015, so a 0.005 threshold is inside the "
    "noise and 3/4 domain wins has p~0.31 under the null. Only the matched full "
    "confirmation can show that uncertainty helps.")


class MissingResults(RuntimeError):
    """A requested domain has no result file for some model."""


def results_dir(model_name, config):
    return (Path(get_data_base_dir()) / config['paths']['results_dir'] / model_name)


def load_model_results(model_name, domains, config):
    """``{domain: ndcg@10}``, raising if ANY requested domain is absent.

    Falling back to the domains that happen to exist would compare macro averages over
    different domain sets, which is worse than not answering.
    """
    base = results_dir(model_name, config)
    out, missing = {}, []
    for d in domains:
        f = base / f"{d}_results.json"
        if not f.exists():
            missing.append(str(f))
            continue
        data = json.loads(f.read_text())
        ndcg = data.get('metrics', {}).get('ndcg_cut_10')
        if ndcg is None:
            missing.append(f"{f} (no ndcg_cut_10)")
            continue
        out[d] = float(ndcg)
    if missing:
        raise MissingResults(
            f"{model_name}: {len(missing)} of {len(domains)} domain results are "
            f"missing:\n    " + "\n    ".join(missing) +
            "\n  Refusing to decide on a partial evaluation.")
    return out


def num_queries(domain):
    q = get_path("processed") / f"{domain}_queries.jsonl"
    if not q.exists():
        return None
    with open(q) as f:
        return sum(1 for line in f if line.strip())


def macro(scores, domains):
    return sum(scores[d] for d in domains) / len(domains)


def compare(baseline, candidate, domains):
    """One candidate vs the lambda=0 control -> verdict dict."""
    base_macro = macro(baseline, domains)
    cand_macro = macro(candidate, domains)
    delta = cand_macro - base_macro
    per_domain = [{
        'domain': d,
        'baseline': baseline[d],
        'candidate': candidate[d],
        'delta': candidate[d] - baseline[d],
        'win': candidate[d] > baseline[d],
        'num_queries': num_queries(d),
    } for d in domains]
    wins = sum(1 for r in per_domain if r['win'])

    # "driven by a single domain": the whole macro gain comes from one domain, so
    # removing it would erase the improvement.
    positives = [r for r in per_domain if r['delta'] > 0]
    single_domain = len(positives) <= 1 and delta > 0

    promotes = delta >= PROMOTE_DELTA - EPS
    inconclusive = delta >= INCONCLUSIVE_DELTA - EPS
    if promotes and wins >= MIN_DOMAIN_WINS and not single_domain:
        verdict, why = 'promote', (
            f"macro delta {delta:+.4f} >= {PROMOTE_DELTA} and won {wins}/{len(domains)} "
            f"domains")
    elif single_domain and inconclusive:
        verdict, why = 'stop', (
            f"macro delta {delta:+.4f} comes from a single domain "
            f"({[r['domain'] for r in positives]}); not a general improvement")
    elif promotes and wins < MIN_DOMAIN_WINS:
        verdict, why = 'stop', (
            f"macro delta {delta:+.4f} but only {wins}/{len(domains)} domain wins "
            f"(need {MIN_DOMAIN_WINS})")
    elif inconclusive:
        verdict, why = 'inconclusive', (
            f"macro delta {delta:+.4f} is in [{INCONCLUSIVE_DELTA}, {PROMOTE_DELTA}); "
            f"run a second pilot seed before committing to a full run")
    else:
        verdict, why = 'stop', (
            f"macro delta {delta:+.4f} < {INCONCLUSIVE_DELTA}; uncertainty is not "
            f"paying for itself")

    return {
        'baseline_macro': base_macro, 'candidate_macro': cand_macro,
        'delta': delta, 'domain_wins': wins, 'num_domains': len(domains),
        'single_domain_driven': single_domain,
        'per_domain': per_domain, 'verdict': verdict, 'reason': why,
    }


def decide(baseline, candidates, domains):
    """Compare every candidate, then apply the smaller-lambda tie-break.

    ``candidates`` is ``{name: {domain: ndcg}}``; ``lambda_of`` is taken from the name
    ordering supplied by the caller (first = lower dose), which is how the pilots are
    submitted.
    """
    results = {name: compare(baseline, scores, domains)
               for name, scores in candidates.items()}
    promoted = [n for n, r in results.items() if r['verdict'] == 'promote']

    chosen, note = None, None
    if len(promoted) == 1:
        chosen = promoted[0]
    elif len(promoted) > 1:
        # ordered low-dose first by the caller; the tie-break prefers the smaller
        # lambda when the two are within TIE_DELTA of each other
        ordered = [n for n in candidates if n in promoted]
        best = max(ordered, key=lambda n: results[n]['delta'])
        spread = abs(results[ordered[0]]['delta'] - results[best]['delta'])
        if spread < TIE_DELTA:
            chosen = ordered[0]
            note = (f"both candidates promote and differ by {spread:.4f} < {TIE_DELTA}; "
                    f"taking the smaller lambda ({chosen})")
        else:
            chosen = best
            note = f"{best} leads by {spread:.4f} >= {TIE_DELTA}"
    return {'per_candidate': results, 'promoted': promoted, 'chosen': chosen,
            'tie_break_note': note, 'screening_caveat': SCREENING_CAVEAT}


def format_report(decision, domains, baseline_name):
    lines = ["=" * 78,
             "  LAMBDA PILOT DECISION — macro NDCG@10 over "
             f"{len(domains)} BRIGHT domains",
             "=" * 78,
             f"  baseline: {baseline_name}", ""]
    for name, r in decision['per_candidate'].items():
        lines.append(f"  {name}")
        lines.append(f"    {'domain':<24}{'lam0':>9}{'cand':>9}{'delta':>9}"
                     f"{'queries':>9}")
        for d in r['per_domain']:
            q = d['num_queries'] if d['num_queries'] is not None else '?'
            lines.append(f"    {d['domain']:<24}{d['baseline']:>9.4f}"
                         f"{d['candidate']:>9.4f}{d['delta']:>+9.4f}{str(q):>9}")
        lines.append(f"    {'MACRO':<24}{r['baseline_macro']:>9.4f}"
                     f"{r['candidate_macro']:>9.4f}{r['delta']:>+9.4f}")
        lines.append(f"    domain wins: {r['domain_wins']}/{r['num_domains']}"
                     + ("  (single-domain driven)" if r['single_domain_driven'] else ""))
        lines.append(f"    VERDICT: {r['verdict'].upper()} — {r['reason']}")
        lines.append("")
    lines.append("-" * 78)
    lines.append(f"  promoted : {decision['promoted'] or 'none'}")
    lines.append(f"  chosen   : {decision['chosen'] or 'none'}")
    if decision['tie_break_note']:
        lines.append(f"  tie-break: {decision['tie_break_note']}")
    lines.append("-" * 78)
    lines.append("  CAVEAT: " + SCREENING_CAVEAT)
    if decision['chosen']:
        lines.append("  Next: rerun lambda=0 AND the chosen lambda with "
                     "--recipe async_fast_grass (max_age_steps=1000), full mixture, "
                     "same seed. The pre-fix lambda=0 run is not a valid control.")
    lines.append("=" * 78)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline', required=True, help='lambda=0 model directory NAME')
    ap.add_argument('--candidates', nargs='+', required=True,
                    help='nonzero-lambda model directory NAMES, LOW dose first')
    ap.add_argument('--domains', required=True,
                    help='comma-separated domains, e.g. biology,economics,...')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    config = load_config()
    domains = [d.strip() for d in args.domains.split(',') if d.strip()]

    try:
        baseline = load_model_results(args.baseline, domains, config)
        candidates = {name: load_model_results(name, domains, config)
                      for name in args.candidates}
    except MissingResults as e:
        print(f"❌ {e}")
        return 2

    decision = decide(baseline, candidates, domains)
    print(format_report(decision, domains, args.baseline))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            'baseline': args.baseline, 'domains': domains, **decision}, indent=2))
        print(f"📄 written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
