import gc
import json
import os
import sys
import pandas as pd
import torch
from pathlib import Path

os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_path, get_training_context, load_config, \
                          encode_to_pickle, build_faiss_index, set_seed
from data.preprocessor import run_setup
from run_grass_mcd import run_mcd_pipeline
from run_grass_seq_bandit import run_seq_bandit_pipeline


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--mode', type=str, default=None,
                        help='Override uncertainty_mode from config: "mc_dropout", "seq_bandit", or "async_v2"')
    parser.add_argument('--n_das', type=int, default=None,
                        help='Override mab_n_das from config (challengers per batch)')
    parser.add_argument('--selection', type=str, default='bandit', choices=['bandit', 'random'],
                        help='[seq_bandit mode] Query selection method')
    parser.add_argument('--coverage', type=float, default=0.25,
                        help='[seq_bandit mode] Fraction of queries to mine per epoch (X in Algorithm 1)')
    parser.add_argument('--lambda_val', type=float, default=None,
                        help='[seq_bandit mode] Override gap-index lambda (default keeps cfg value)')
    parser.add_argument('--build_index_only', action='store_true',
                        help='Build stale ANN index then exit — run this before parallel sweeps')
    parser.add_argument('--model_suffix', type=str, default=None,
                        help='Append suffix to model output dir (e.g. "ndas5") to avoid collisions')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Override num_epochs from config')
    cli_args, _ = parser.parse_known_args()

    corpus_file, query_file, qrels_file = run_setup()

    ctx    = get_training_context("grass")
    config = load_config()
    cfg    = config['training']['grass']
    set_seed(config.get('seed', 42))

    uncertainty_mode = cli_args.mode or cfg.get('uncertainty_mode', 'mc_dropout')
    if uncertainty_mode == 'async_v2':
        # Hand off to the 2-GPU async v2 orchestrator. Strip --mode from argv so the
        # v2 main() can re-parse cleanly with strict argparse.
        filtered, skip_next = [], False
        for tok in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if tok == '--mode':
                skip_next = True
                continue
            if tok.startswith('--mode='):
                continue
            filtered.append(tok)
        sys.argv = [sys.argv[0]] + filtered
        from train_grass_async_v2 import main as v2_main
        v2_main()
        return

    workdir = get_path("temp_grass")
    workdir.mkdir(exist_ok=True, parents=True)

    if cli_args.n_das is not None:
        cfg = {**cfg, 'mab_n_das': cli_args.n_das}
        print(f"  CLI override: mab_n_das={cli_args.n_das}", flush=True)

    if cli_args.model_suffix is not None:
        cfg = {**cfg, 'model_name': cfg['model_name'] + '_' + cli_args.model_suffix}
        print(f"  CLI override: model_name={cfg['model_name']}", flush=True)

    if cli_args.num_epochs is not None:
        cfg = {**cfg, 'num_epochs': cli_args.num_epochs}
        print(f"  CLI override: num_epochs={cli_args.num_epochs}", flush=True)

    if cli_args.lambda_val is not None:
        cfg = {**cfg, 'lambda_val': cli_args.lambda_val}
        print(f"  CLI override: lambda_val={cli_args.lambda_val}", flush=True)

    stale_dir = workdir / "stale_index"
    stale_dir.mkdir(exist_ok=True)
    stale_pkl = stale_dir / "corpus.pkl"
    if not stale_pkl.exists():
        print("📦 Building stale ANN index from base model...", flush=True)
        encode_to_pickle(cfg['base_model'], corpus_file, stale_pkl, False, ctx, config)
    stale_idx, stale_embs, c_ids = build_faiss_index(stale_pkl)
    c_id_to_idx = {did: i for i, did in enumerate(c_ids)}
    print(f"✅ Stale index ready: {len(c_ids)} passages", flush=True)

    if cli_args.build_index_only:
        print("✅ Index built. Exiting.", flush=True)
        return

    corpus_lookup = {}
    with open(corpus_file) as f:
        for line in f:
            d = json.loads(line)
            corpus_lookup[d['docid']] = d['text']
    print(f"Loaded corpus lookup: {len(corpus_lookup)} passages", flush=True)

    qrels_data = []
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4: qrels_data.append({'qid': parts[0], 'did': parts[2]})
    qrels_dict = pd.DataFrame(qrels_data).groupby('qid')['did'].apply(set).to_dict()

    mix_df = pd.read_json(query_file, lines=True)

    if cli_args.debug:
        mix_df = mix_df.head(100)
        cfg = {**cfg, 'T': 2, 'P': 20, 'L': 5, 'm': 2, 'query_batch_size': 10}
        print("🐛 DEBUG mode: 100 queries, T=2, P=20, L=5", flush=True)

    print(f"✅ Setup complete. corpus_lookup={len(corpus_lookup)} passages, "
          f"qrels={len(qrels_dict)} queries, mix_df={len(mix_df)} unique queries.", flush=True)

    uncertainty_mode = cli_args.mode or cfg.get('uncertainty_mode', 'mc_dropout')
    print(f"\n{'='*50}", flush=True)
    print(f"  GRASS MODE: {uncertainty_mode.upper()}", flush=True)
    print(f"{'='*50}\n", flush=True)

    if uncertainty_mode == 'seq_bandit':
        model_suffix = cli_args.model_suffix or ''
        num_epochs   = cli_args.num_epochs if cli_args.num_epochs is not None else cfg.get('num_epochs', 3)
        output_model_dir = run_seq_bandit_pipeline(
            stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup, mix_df,
            qrels_dict, cfg, config, ctx, workdir,
            selection=cli_args.selection,
            coverage=cli_args.coverage,
            num_epochs=num_epochs,
            model_suffix=model_suffix,
        )
    else:
        output_model_dir = run_mcd_pipeline(
            stale_idx, stale_embs, c_id_to_idx, c_ids, corpus_lookup, mix_df,
            qrels_dict, cfg, config, ctx, workdir
        )

    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ GRASS complete. Model saved to: {output_model_dir}", flush=True)


if __name__ == "__main__":
    main()
