"""BM25 baseline evaluation across all BRIGHT domains using Pyserini (Lucene).

Indexes are provenance-aware. Each index carries an ``index_meta.json`` recording the
corpus digest and the pyserini version that built it; reuse requires an exact match,
and anything else is archived and rebuilt. Existence alone is not evidence: a
truncated index from a killed job is a non-empty directory, and a corpus regenerated
under different preprocessing leaves the old index silently in place.
"""

import os
import shutil
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Resolve project root and add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import (load_config, get_path, get_data_base_dir,
                           load_excluded_ids, search_depth, apply_exclusions,
                           _load_qrels, check_eval_artifacts, _sha256, atomic_write,
                           _package_versions, eval_artifact_hashes, retry_io,
                           _load_corpus_lookup)
from evaluation.trec_eval_wrapper import TrecEvalWrapper


META_NAME = "index_meta.json"


def check_and_prepare_bm25_data(domains: list, index_base: Path,
                                processed_dir: Path = None) -> None:
    """Derive each domain's Pyserini corpus from the dense evaluation corpus.

    Single source of truth. This used to reload BRIGHT independently and skip on mere
    file existence, so the sparse and dense arms could end up scoring two different
    collections and the comparison between them would be meaningless. The dense
    `{domain}_corpus.jsonl` is what evaluation.md's protocol is defined over, so it is
    what BM25 is built from, and an existing BM25 corpus is validated against it by
    exact docid -> text equality rather than being trusted.
    """
    processed_dir = Path(processed_dir) if processed_dir else get_path("processed")

    for domain in domains:
        corpus_dir = index_base / domain / "corpus"
        corpus_file = corpus_dir / "corpus.jsonl"
        dense_corpus = processed_dir / f"{domain}_corpus.jsonl"
        dense_qrels = processed_dir / f"{domain}_qrels.txt"

        if not dense_corpus.is_file():
            raise RuntimeError(
                f"[{domain}] {dense_corpus} is missing; BM25 is derived from the dense "
                f"evaluation corpus so the two arms score the same collection. "
                f"Run `python src/data/preprocessor.py` first.")

        if corpus_file.is_file() and bm25_corpus_matches_dense(corpus_file, dense_corpus):
            print(f"  [{domain}] Pyserini corpus matches the dense corpus.")
        else:
            why = "missing" if not corpus_file.is_file() else "differs from the dense corpus"
            print(f"  [{domain}] Pyserini corpus {why}; regenerating from "
                  f"{dense_corpus.name}...")
            corpus_dir.mkdir(parents=True, exist_ok=True)
            with atomic_write(corpus_file) as out:
                with open(dense_corpus, encoding='utf-8') as src:
                    for line in src:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        out.write(json.dumps(
                            {"id": str(row["docid"]), "contents": row.get("text") or ""},
                            ensure_ascii=False) + "\n")

        if dense_qrels.is_file():
            verify_qrel_documents(domain, corpus_file, dense_qrels)


def corpus_provenance(corpus_file: Path) -> dict:
    """What an index must match to be reusable.

    k1/b are deliberately absent: Pyserini applies them at search time via
    set_bm25(), so changing them does not invalidate an index.
    """
    return {
        "corpus_sha256": _sha256(corpus_file),
        "corpus_bytes": corpus_file.stat().st_size,
        "pyserini_version": _package_versions(("pyserini",))["pyserini"],
    }


def build_lucene_index(corpus_dir: Path, index_dir: Path, threads: int = 4) -> None:
    """Run Pyserini Lucene indexer via subprocess."""
    index_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'pyserini.index.lucene',
        '--collection', 'JsonCollection',
        '--input', str(corpus_dir),
        '--index', str(index_dir),
        '--generator', 'DefaultLuceneDocumentGenerator',
        '--threads', str(threads),
    ]
    print(f"  Building Lucene index at {index_dir} ...")
    subprocess.run(cmd, check=True)


def ensure_index(corpus_file: Path, corpus_dir: Path, index_dir: Path,
                 *, allow_rebuild: bool = True) -> dict:
    """Reuse the index only if its recorded provenance still holds.

    Rebuild-by-default with an explicit opt-out, matching refresh_stale_index.py.
    A non-empty index directory with no meta is a partial build from a killed job,
    not a usable index, so it is rebuilt rather than searched.
    """
    provenance = corpus_provenance(corpus_file)
    meta_path = index_dir.parent / META_NAME
    prior = None
    if meta_path.is_file():
        try:
            prior = json.loads(meta_path.read_text())
        except (ValueError, OSError):
            prior = {}

    populated = index_dir.exists() and any(index_dir.iterdir())
    stale = [k for k, v in provenance.items() if (prior or {}).get(k) != v]

    if populated and prior is not None and not stale:
        print(f"  Index provenance matches (corpus {provenance['corpus_sha256'][:12]}, "
              f"pyserini {provenance['pyserini_version']}); reusing.")
        return provenance

    reason = ("no index" if not populated
              else "no index_meta.json (partial build?)" if prior is None
              else f"changed: {', '.join(stale)}")
    if not allow_rebuild:
        raise RuntimeError(
            f"{index_dir} cannot be reused ({reason}) and --no-rebuild was given. "
            f"Re-run without it to archive and rebuild.")

    # Staged publication. Building straight into index_dir left a half-written index
    # at the canonical path when a job was killed, and because the OLD meta was still
    # beside it, an unchanged corpus made that partial index match its provenance and
    # be reported as "reusing". Build aside, publish atomically, write meta last.
    staging = index_dir.with_name(f"{index_dir.name}.building")
    if staging.exists():
        retry_io(lambda: shutil.rmtree(staging), f"remove stale staging {staging}")
    if staging.exists():
        raise RuntimeError(
            f"could not remove stale BM25 staging directory {staging}. Building "
            f"into it could mix files from different corpora, so nothing will be "
            f"published. Remove it by hand and retry.")
    try:
        build_lucene_index(corpus_dir, staging)
    except BaseException:
        retry_io(lambda: shutil.rmtree(staging), f"remove failed staging {staging}")
        raise

    if populated:
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        archived = index_dir.with_name(f"{index_dir.name}.old_{stamp}")
        index_dir.rename(archived)
        # The archived index keeps its own provenance. Unlinking it would leave an
        # index nothing can identify, which is the state this whole function exists
        # to prevent.
        if meta_path.is_file():
            meta_path.rename(archived / META_NAME)
        print(f"  Index unusable ({reason}); archived -> {archived.name}")

    os.replace(staging, index_dir)
    with atomic_write(meta_path) as f:
        json.dump({**provenance, "built_at": datetime.now().isoformat()}, f, indent=2)
    return provenance


def bm25_corpus_matches_dense(bm25_corpus: Path, dense_corpus: Path) -> bool:
    """Exact docid -> text equality between the BM25 and dense views of a domain.

    Docid-set equality is not enough: the same id carrying different text is a
    different corpus, and the sparse/dense comparison would silently be run over two
    different collections.
    """
    def load_pyserini(path):
        """The BM25 view only: {id, contents}. The dense view has its own reader."""
        out = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    out[str(row["id"])] = row.get("contents") or ""
        return out

    if not Path(bm25_corpus).is_file() or not Path(dense_corpus).is_file():
        return False
    # _load_corpus_lookup is the shared reader for the dense {docid, text} schema.
    dense = {str(k): (v or "") for k, v in _load_corpus_lookup(dense_corpus).items()}
    return load_pyserini(bm25_corpus) == dense


def verify_qrel_documents(domain: str, corpus_file: Path, qrels_file: Path) -> None:
    """Every judged document must exist in the corpus, or the domain is unscoreable."""
    ids = set()
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                ids.add(str(json.loads(line)["id"]))
    # _load_qrels is the one strict reader for mining AND evaluation: it raises on a
    # row that is not exactly four columns. Parsing here by hand silently accepted
    # malformed three-column rows that it rejects.
    judged = set().union(*_load_qrels(qrels_file).values())
    missing = {d for d in judged if d not in ids}
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise RuntimeError(
            f"[{domain}] {len(missing)} judged document(s) are absent from the BM25 "
            f"corpus (e.g. {sample}). They can never be retrieved, so recall and "
            f"NDCG for this domain would be understated against dense retrieval.")


def preflight_java() -> str:
    """Fail on Java before importing Pyserini or touching an index.

    Pyserini needs a JVM. Without this the failure surfaces as an opaque error deep
    inside indexing, after the corpus work is already done. Defect P9: the launcher
    pointed at a /scratch JDK that does not exist.
    """
    java_home = os.environ.get("JAVA_HOME", "").strip()
    if not java_home or not Path(java_home).is_dir():
        raise RuntimeError(
            f"JAVA_HOME is not set to an existing directory (got {java_home!r}). "
            f"Pyserini/Lucene needs a JDK 11+; see run_bm25_singularity.sh.")
    java_bin = Path(java_home) / "bin" / "java"
    if not (java_bin.is_file() and os.access(java_bin, os.X_OK)):
        raise RuntimeError(f"{java_bin} is missing or not executable (bin/java).")
    if not list(Path(java_home).rglob("libjvm.so")):
        raise RuntimeError(f"no libjvm.so under {java_home}; the JDK is incomplete.")
    try:
        proc = subprocess.run([str(java_bin), "-version"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"could not run {java_bin} -version: {e}")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{java_bin} -version exited {proc.returncode}: "
            f"{proc.stderr.decode(errors='replace')[:300]}")
    return java_home


def load_queries(queries_file: Path):
    """Load queries from Tevatron-format JSONL as (query_id, query_text) tuples."""
    queries = []
    with open(queries_file, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            qid = str(d['query_id'])
            qtext = d.get('query') or d.get('text', '')
            queries.append((qid, qtext))
    return queries


def main():
    parser = argparse.ArgumentParser(description="BM25 baseline over BRIGHT.")
    parser.add_argument('--no-rebuild', action='store_true',
                        help="Fail instead of rebuilding when an index is stale or partial.")
    args = parser.parse_args()

    config = load_config()
    processed_dir = get_path("processed")
    results_dir = get_path("results")
    index_base = get_data_base_dir() / "bm25_indices"
    results_dir.mkdir(parents=True, exist_ok=True)

    bm25_cfg = config['evaluation'].get('bm25', {})
    k1 = bm25_cfg.get('k1', 0.9)
    b = bm25_cfg.get('b', 0.4)
    top_k = config['evaluation']['top_k']
    domains = config['evaluation']['eval_domains']
    # Distinct per parameterisation, so two BM25 sweeps cannot overwrite each other's
    # summary -- the same reason the dense path keys results by model_run_tag.
    run_tag = f"bm25_k1-{k1}_b-{b}"

    # Pyserini initializes the JVM while importing LuceneSearcher, so validate the
    # JDK first or a broken JAVA_HOME bypasses the actionable preflight below.
    print(f"Java: {preflight_java()}")

    try:
        from pyserini.search.lucene import LuceneSearcher
    except Exception as e:
        print(f"ERROR: Could not import pyserini.\n  {e}")
        print(
            "Ensure:\n"
            "  1. pyserini is installed: pip install --user pyserini\n"
            "  2. JAVA_HOME points to JDK 11+ (see one-time setup in run_bm25_singularity.sh)"
        )
        sys.exit(1)

    print(f"BM25 Evaluation  k1={k1}  b={b}  top_k={top_k}")
    print(f"Domains ({len(domains)}): {domains}\n")

    # --- Data preparation: derived from the dense corpus, same collection ---
    print("Checking/preparing Pyserini corpus files...")
    check_and_prepare_bm25_data(domains, index_base)
    print()

    summary = {}
    per_domain = []
    failed = []
    for domain in domains:
        print(f"--- {domain} ---")
        queries_file  = processed_dir / f"{domain}_queries.jsonl"
        qrels_file    = processed_dir / f"{domain}_qrels.txt"
        excluded_file = processed_dir / f"{domain}_excluded.json"
        corpus_dir    = index_base / domain / "corpus"
        corpus_file   = corpus_dir / "corpus.jsonl"
        index_dir     = index_base / domain / "index"

        # Query, qrel and exclusion files must exist (generated by dense eval prep).
        # A domain evaluated without its exclusions would silently report the old,
        # unfiltered numbers, so a missing file fails the domain rather than skipping.
        missing = [f for f in (queries_file, qrels_file, excluded_file) if not f.exists()]
        if missing:
            for f in missing:
                print(f"  Missing: {f}")
            print(f"  FAIL: run the dense eval preprocessor first.\n")
            failed.append(domain)
            continue

        try:
            # Reuse only a provably current index; otherwise archive and rebuild.
            provenance = ensure_index(corpus_file, corpus_dir, index_dir,
                                      allow_rebuild=not args.no_rebuild)

            # Search
            searcher = LuceneSearcher(str(index_dir))
            searcher.set_bm25(k1=k1, b=b)
            corpus_size = sum(1 for line in open(corpus_file, encoding='utf-8') if line.strip())

            queries = load_queries(queries_file)
            # Same exclusion rule as the dense path: over-retrieve per query, then
            # filter, so the excluded documents do not eat top_k slots.
            excluded = load_excluded_ids(domain, processed_dir)
            qrels = _load_qrels(qrels_file)
            # BM25 encodes nothing, so no encoded ids to check -- but the query,
            # qrel and exclusion sets must still agree or the metric moves silently.
            check_eval_artifacts(domain, qrels, excluded,
                                 query_ids=[qid for qid, _ in queries])
            run_results = {}
            for qid, qtext in queries:
                # Clamped like the dense paths (evaluate.py, helpers.evaluate_bright):
                # aops excludes up to 11,224 docs for one query, so an unclamped depth
                # allocates a priority queue far larger than the corpus.
                depth = min(search_depth(top_k, excluded, qid), corpus_size)
                hits = searcher.search(qtext, k=depth)
                run_results[qid] = {hit.docid: hit.score for hit in hits}
            run_results = apply_exclusions(run_results, excluded, top_k)
            print(f"  Searched {len(queries)} queries.")

            # Evaluate
            evaluator = TrecEvalWrapper(qrels)
            metrics = evaluator.evaluate(run_results, {'recip_rank', 'ndcg_cut_10', 'recall_1000'})

            # Save (same schema as dense eval results)
            result = {
                "domain": domain,
                "model_path": f"BM25 (k1={k1}, b={b})",
                "run_tag": run_tag,
                "bm25": {"k1": k1, "b": b},
                "metrics": metrics,
                **provenance,
            }
            out_file = results_dir / f"{domain}_results_bm25.json"
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2)
            per_domain.append({"domain": domain, "num_queries": len(queries), **metrics})

            mrr  = metrics.get('recip_rank', 0.0)
            ndcg = metrics.get('ndcg_cut_10', 0.0)
            rec  = metrics.get('recall_1000', 0.0)
            print(f"  MRR={mrr:.4f}  NDCG@10={ndcg:.4f}  Recall@1000={rec:.4f}")
            print(f"  Saved -> {out_file}\n")
            summary[domain] = metrics

        except Exception as e:
            print(f"  ERROR on {domain}: {e}")
            import traceback; traceback.print_exc()
            print()
            failed.append(domain)
            continue

    # Final summary table
    if summary:
        print("=" * 62)
        print("SUMMARY  BM25")
        print(f"{'Domain':<28} {'MRR':>8} {'NDCG@10':>9} {'R@1000':>8}")
        print("-" * 62)
        for dom, m in summary.items():
            print(
                f"{dom:<28} {m.get('recip_rank', 0):.4f}  "
                f"{m.get('ndcg_cut_10', 0):.4f}  {m.get('recall_1000', 0):.4f}"
            )
        print("=" * 62)

    if failed:
        # Collect every domain first, but never report success for a partial run:
        # a job that evaluated 9 of 12 domains is not a BM25 baseline.
        print(f"\n❌ {len(failed)} of {len(domains)} domains failed: {', '.join(failed)}")
        return 1

    # Same schema as run_all_evals.py's dense summary, and written only for a complete
    # run, so `--compare_bm25` can require the two domain sets to agree.
    macro = sum(r['ndcg_cut_10'] for r in per_domain) / len(per_domain)
    summary_path = results_dir / run_tag / "summary.json"
    with atomic_write(summary_path) as f:
        json.dump({
            'model': f"BM25 (k1={k1}, b={b})",
            'model_name': 'bm25',
            'run_tag': run_tag,
            'domains': domains,
            'bm25': {'k1': k1, 'b': b},
            'per_domain': per_domain,
            'macro_ndcg_cut_10': macro,
            # What --compare_bm25 checks alongside the domain set: same domains over
            # regenerated corpora is still not a comparable pair of macro scores.
            'eval_artifact_sha256': eval_artifact_hashes(get_path("processed"), domains),
        }, f, indent=2)
    print(f"📄 Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
