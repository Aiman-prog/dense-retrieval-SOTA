"""Evaluate trained models using Tevatron CLI commands."""

import sys
import argparse
import subprocess
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent.parent

# Import using absolute paths
from src.data.bright_loader import BRIGHTLoader
from src.data.preprocessor import BRIGHTPreprocessor
from src.utils.helpers import load_config, get_data_base_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models using Tevatron CLI")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model (Tevatron checkpoint)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="biology",
        help="Domain to evaluate on (default: biology)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k retrieval depth (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for encoding (default: 512)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(str(project_root / 'config' / 'config.yaml'))
    base_dir = Path(get_data_base_dir())
    
    # Setup paths
    processed_dir = base_dir / 'data' / 'processed'
    eval_dir = base_dir / 'data' / 'evaluation' / args.domain
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n{'='*80}")
    print(f"Step 1: Loading BRIGHT dataset for domain: {args.domain}")
    print(f"{'='*80}")
    loader = BRIGHTLoader(config_path=str(project_root / 'config' / 'config.yaml'))
    loader.load_dataset()
    
    # Prepare Tevatron format files
    print(f"\n{'='*80}")
    print(f"Step 2: Preparing Tevatron format files")
    print(f"{'='*80}")
    preprocessor = BRIGHTPreprocessor(output_dir=str(processed_dir))
    
    corpus_df = loader.get_corpus(args.domain)
    queries_df = loader.get_queries(args.domain)
    qrels_df = loader.get_qrels(args.domain)
    
    corpus_file = preprocessor.prepare_tevatron_corpus(
        corpus_df, 
        filename=f"{args.domain}_corpus.jsonl"
    )
    queries_file = preprocessor.prepare_tevatron_queries(
        queries_df,
        filename=f"{args.domain}_queries.jsonl"
    )
    qrels_file = preprocessor.prepare_trec_qrels(
        qrels_df,
        filename=f"{args.domain}_qrels.txt"
    )
    
    print(f"✅ Corpus: {corpus_file}")
    print(f"✅ Queries: {queries_file}")
    print(f"✅ Qrels: {qrels_file}")
    
    # Encode corpus
    print(f"\n{'='*80}")
    print(f"Step 3: Encoding corpus using Tevatron")
    print(f"{'='*80}")
    corpus_emb_dir = eval_dir / 'corpus_emb'
    corpus_emb_dir.mkdir(exist_ok=True)
    
    corpus_emb_file = corpus_emb_dir / 'corpus.pkl'
    encode_corpus_cmd = [
        sys.executable, '-m', 'tevatron.driver.encode',
        '--output_dir', str(corpus_emb_dir),
        '--model_name_or_path', args.model_path,
        '--fp16',
        '--per_device_eval_batch_size', str(args.batch_size),
        '--encode_in_path', corpus_file,
        '--encoded_save_path', str(corpus_emb_file)
    ]
    
    print(f"Running: {' '.join(encode_corpus_cmd)}")
    result = subprocess.run(encode_corpus_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: Corpus encoding failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, encode_corpus_cmd, result.stdout, result.stderr)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Check if embedding file was created
    if not corpus_emb_file.exists():
        raise FileNotFoundError(f"Corpus embeddings file not created: {corpus_emb_file}")
    
    print(f"✅ Corpus embeddings saved to: {corpus_emb_file}")
    
    # Encode queries
    print(f"\n{'='*80}")
    print(f"Step 4: Encoding queries using Tevatron")
    print(f"{'='*80}")
    query_emb_dir = eval_dir / 'query_emb'
    query_emb_dir.mkdir(exist_ok=True)
    
    query_emb_file = query_emb_dir / 'query.pkl'
    encode_query_cmd = [
        sys.executable, '-m', 'tevatron.driver.encode',
        '--output_dir', str(query_emb_dir),
        '--model_name_or_path', args.model_path,
        '--fp16',
        '--per_device_eval_batch_size', str(args.batch_size),
        '--encode_in_path', queries_file,
        '--encoded_save_path', str(query_emb_file)
    ]
    
    print(f"Running: {' '.join(encode_query_cmd)}")
    result = subprocess.run(encode_query_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: Query encoding failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, encode_query_cmd, result.stdout, result.stderr)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Check if embedding file was created
    if not query_emb_file.exists():
        raise FileNotFoundError(f"Query embeddings file not created: {query_emb_file}")
    
    print(f"✅ Query embeddings saved to: {query_emb_file}")
    
    # Retrieve
    print(f"\n{'='*80}")
    print(f"Step 5: Retrieving top-{args.k} documents using Tevatron FAISS")
    print(f"{'='*80}")
    ranking_file = eval_dir / 'ranking.tsv'
    
    retrieve_cmd = [
        sys.executable, '-m', 'tevatron.faiss_retriever',
        '--query_reps', str(query_emb_file),
        '--passage_reps', str(corpus_emb_file),
        '--depth', str(args.k),
        '--batch_size', str(args.batch_size),
        '--save_ranking_to', str(ranking_file)
    ]
    
    print(f"Running: {' '.join(retrieve_cmd)}")
    result = subprocess.run(retrieve_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: Retrieval failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, retrieve_cmd, result.stdout, result.stderr)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if not ranking_file.exists():
        raise FileNotFoundError(f"Ranking file not created: {ranking_file}")
    
    # Check file size and format
    file_size = ranking_file.stat().st_size
    print(f"✅ Rankings saved to: {ranking_file} (size: {file_size} bytes)")
    
    if file_size == 0:
        raise ValueError(f"Ranking file is empty: {ranking_file}")
    
    # Check if file is text or binary
    try:
        with open(ranking_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            print(f"First line preview: {first_line[:100]}")
    except UnicodeDecodeError:
        print("⚠️  Warning: Ranking file appears to be binary or non-UTF-8 encoded")
    
    # Evaluate using trec_eval (via pyserini or direct trec_eval)
    print(f"\n{'='*80}")
    print(f"Step 6: Evaluating retrieval results")
    print(f"{'='*80}")
    
    # Parse Tevatron ranking output (saved as pickle file)
    import pickle
    
    # Tevatron saves rankings as pickle file, not TSV
    with open(ranking_file, 'rb') as f:
        rankings = pickle.load(f)
    
    # Convert to dict format: {query_id: {doc_id: score}}
    run_results = {}
    if isinstance(rankings, list):
        # Format: [(qid, doc_id, score), ...]
        for item in rankings:
            if len(item) >= 3:
                qid = str(item[0])
                doc_id = str(item[1])
                score = float(item[2])
                if qid not in run_results:
                    run_results[qid] = {}
                run_results[qid][doc_id] = score
    elif isinstance(rankings, dict):
        # Already in dict format or different structure
        run_results = rankings
    
    if not run_results:
        raise ValueError(f"Could not parse ranking file: {ranking_file}. Unexpected format.")
    
    # Use trec_eval wrapper
    from src.evaluation.trec_eval_wrapper import TrecEvalWrapper
    
    evaluator = TrecEvalWrapper(qrels_df)
    metrics = evaluator.evaluate(
        run_results=run_results,
        metrics={'recip_rank', 'ndcg_cut_10', 'recall_10'}
    )
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS for {args.domain}")
    print(f"{'='*80}")
    print(f"MRR: {metrics.get('recip_rank', 0):.4f}")
    print(f"NDCG@10: {metrics.get('ndcg_cut_10', 0):.4f}")
    print(f"Recall@10: {metrics.get('recall_10', 0):.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
