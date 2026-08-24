"""BRIGHT evaluation runner: one `src/evaluation/evaluate.py` subprocess per domain.

Two behaviours matter for the lambda pilot and are easy to get wrong:

* ``--domains`` restricts the run to a subset (the pilot evaluates four development
  domains, not all twelve);
* a failed domain makes the whole run exit NONZERO. This loop used to catch
  ``CalledProcessError`` and continue, so a job that evaluated two of four domains
  still reported success and the decision tool would then compare partial results.

``--require_existing`` turns data preparation into verification. By default a missing
domain is prepared from HuggingFace; for the pilot that would silently regenerate
processed data mid-experiment, so the flag makes it an error instead.
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

# Resolve project root and add to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

# Import your helpers and classes
from utils.helpers import load_config, get_data_base_dir, get_path
from data.preprocessor import BRIGHTPreprocessor
from data.bright_loader import BRIGHTLoader


def domain_files(domain):
    processed_dir = get_path("processed")
    return [
        processed_dir / f"{domain}_corpus.jsonl",
        processed_dir / f"{domain}_queries.jsonl",
        processed_dir / f"{domain}_qrels.txt",
        processed_dir / f"{domain}_excluded.json",
    ]


def check_and_prepare_data(domains, config, require_existing=False):
    """Verify each domain's processed files; prepare them unless forbidden.

    ``require_existing=True`` returns the list of domains whose files are missing
    instead of building them — the pilot must never regenerate processed data as a
    side effect of an evaluation job.
    """
    loader = None
    preprocessor = BRIGHTPreprocessor()
    missing_domains = []

    for domain in domains:
        required_files = domain_files(domain)

        if not all(f.exists() for f in required_files):
            if require_existing:
                absent = [str(f) for f in required_files if not f.exists()]
                print(f"❌ Data for '{domain}' is missing and --require_existing is "
                      f"set; refusing to regenerate: {absent}")
                missing_domains.append(domain)
                continue
            print(f"📦 Data for '{domain}' missing in {get_path('processed')}. "
                  f"Processing...")
            if loader is None:
                loader = BRIGHTLoader()
                loader.load_dataset()

            domain_data = loader.get_data_split(domain)
            preprocessor.prepare_tevatron_corpus(domain_data['corpus'], f"{domain}_corpus.jsonl")
            preprocessor.prepare_tevatron_queries(domain_data['queries'], f"{domain}_queries.jsonl")
            preprocessor.prepare_trec_qrels(domain_data['qrels'], f"{domain}_qrels.txt")
            preprocessor.prepare_bright_excluded(domain_data['excluded'],
                                                 f"{domain}_excluded.json")
        else:
            print(f"✅ Data for '{domain}' verified.")
    return missing_domains


def _num_queries(domain):
    """Query count, so a macro average is reported next to the sample it rests on."""
    q_file = get_path("processed") / f"{domain}_queries.jsonl"
    if not q_file.exists():
        return None
    with open(q_file) as f:
        return sum(1 for line in f if line.strip())


def collect_results(model_path, domains, config):
    """Read back the per-domain JSONs `evaluate.py` wrote; returns (rows, missing)."""
    base = Path(get_data_base_dir()) / config['paths']['results_dir'] / Path(model_path).name
    rows, missing = [], []
    for domain in domains:
        f = base / f"{domain}_results.json"
        if not f.exists():
            missing.append(domain)
            continue
        d = json.loads(f.read_text())
        rows.append({
            'domain': domain,
            'ndcg_cut_10': d.get('metrics', {}).get('ndcg_cut_10'),
            'recip_rank': d.get('metrics', {}).get('recip_rank'),
            'recall_1000': d.get('metrics', {}).get('recall_1000'),
            'num_queries': _num_queries(domain),
        })
    return rows, missing


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, help="Path to model")
    parser.add_argument("--domains", type=str, default=None,
                        help="comma-separated subset (default: evaluation.eval_domains)")
    parser.add_argument("--results_json", type=str, default=None,
                        help="write a per-domain + macro NDCG@10 summary here")
    parser.add_argument("--require_existing", action="store_true",
                        help="fail instead of preparing missing BRIGHT domain files")
    args = parser.parse_args()

    # 1. Resolve Model Path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        base_dir = Path(get_data_base_dir())
        model_name = config['training']['crossbatch']['model_name']
        model_path = base_dir / 'models' / model_name

    # 2. Resolve Eval Script Path (The Fix)
    eval_script = Path(__file__).parent.parent / "src" / "evaluation" / "evaluate.py"

    print(f"🕵️  Starting Evaluation Runner")
    print(f"🏗️  Model: {model_path}")
    print(f"📄 Script: {eval_script}\n")

    # Final Safety Checks
    if not model_path.exists():
        print(f"❌ ERROR: Model path does not exist: {model_path}")
        sys.exit(1)
    if not eval_script.exists():
        print(f"❌ ERROR: Evaluation script not found at {eval_script}")
        sys.exit(1)

    all_domains = config['evaluation'].get('eval_domains', [])
    if args.domains:
        domains = [d.strip() for d in args.domains.split(',') if d.strip()]
        unknown = [d for d in domains if d not in all_domains]
        if unknown:
            print(f"❌ ERROR: unknown domain(s) {unknown}; known: {all_domains}")
            sys.exit(1)
    else:
        domains = all_domains
    print(f"🌐 Domains ({len(domains)}): {', '.join(domains)}\n")

    # Check/Prepare data before loop
    missing = check_and_prepare_data(domains, config,
                                     require_existing=args.require_existing)

    failed = list(missing)
    for domain in domains:
        if domain in missing:
            continue
        print(f"\n--- 🌐 Evaluating Domain: {domain} ---")
        cmd = [
            sys.executable, str(eval_script),
            "--model_path", str(model_path),
            "--domain", domain,
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            # Recorded, NOT swallowed: a partial evaluation must not look successful,
            # or a downstream comparison would silently rest on fewer domains.
            print(f"❌ Domain {domain} failed.")
            failed.append(domain)
            continue

    rows, absent = collect_results(model_path, [d for d in domains if d not in failed],
                                   config)
    scored = [r for r in rows if r['ndcg_cut_10'] is not None]
    macro = (sum(r['ndcg_cut_10'] for r in scored) / len(scored)) if scored else None

    print("\n" + "=" * 60)
    print(f"  {Path(model_path).name}")
    print("=" * 60)
    for r in rows:
        print(f"  {r['domain']:<22} NDCG@10={r['ndcg_cut_10']:.4f}  "
              f"({r['num_queries']} queries)")
    if macro is not None:
        print(f"  {'MACRO NDCG@10':<22} {macro:.4f}  over {len(scored)} domains")
    print("=" * 60)

    if args.results_json:
        out = Path(args.results_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            'model': str(model_path),
            'model_name': Path(model_path).name,
            'domains': domains,
            'per_domain': rows,
            'macro_ndcg_cut_10': macro,
            'failed_domains': failed,
            'missing_result_files': absent,
        }, indent=2))
        print(f"📄 Summary written to {out}")

    if failed or absent:
        print(f"\n❌ Evaluation INCOMPLETE — failed: {failed}, "
              f"missing results: {absent}")
        sys.exit(1)

    print("\n🏁 All evaluations complete.")


if __name__ == "__main__":
    main()
