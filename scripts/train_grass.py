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
from run_grass_ema import train_with_ema_grass
from run_grass_mcd import run_mcd_pipeline


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--mode', type=str, default=None,
                        help='Override uncertainty_mode from config: "ema" or "mc_dropout"')
    parser.add_argument('--n_das', type=int, default=None,
                        help='Override mab_n_das from config (challengers per batch)')
    cli_args, _ = parser.parse_known_args()

    corpus_file, query_file, qrels_file = run_setup()

    corpus_lookup = {}
    with open(corpus_file) as f:
        for line in f:
            d = json.loads(line)
            corpus_lookup[d['docid']] = d['text']
    print(f"Loaded corpus lookup: {len(corpus_lookup)} passages", flush=True)

    ctx    = get_training_context("grass")
    config = load_config()
    cfg    = config['training']['grass']
    set_seed(config.get('seed', 42))

    workdir = get_path("temp_grass")
    workdir.mkdir(exist_ok=True, parents=True)

    qrels_data = []
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4: qrels_data.append({'qid': parts[0], 'did': parts[2]})
    qrels_dict = pd.DataFrame(qrels_data).groupby('qid')['did'].apply(set).to_dict()

    mix_df = pd.read_json(query_file, lines=True)

    if cli_args.n_das is not None:
        cfg = {**cfg, 'mab_n_das': cli_args.n_das}
        print(f"  CLI override: mab_n_das={cli_args.n_das}", flush=True)

    if cli_args.debug:
        mix_df = mix_df.head(100)
        cfg = {**cfg, 'T': 2, 'P': 20, 'L': 5, 'm': 2, 'query_batch_size': 10}
        print("🐛 DEBUG mode: 100 queries, T=2, P=20, L=5", flush=True)

    print(f"✅ Setup complete. corpus_lookup={len(corpus_lookup)} passages, "
          f"qrels={len(qrels_dict)} queries, mix_df={len(mix_df)} unique queries.", flush=True)

    stale_dir = workdir / "stale_index"
    stale_dir.mkdir(exist_ok=True)
    stale_pkl = stale_dir / "corpus.pkl"
    if not stale_pkl.exists():
        print("📦 Building stale ANN index from base model...", flush=True)
        encode_to_pickle(cfg['base_model'], corpus_file, stale_pkl, False, ctx, config)
    stale_idx, stale_embs, c_ids = build_faiss_index(stale_pkl)
    c_id_to_idx = {did: i for i, did in enumerate(c_ids)}
    print(f"✅ Stale index ready: {len(c_ids)} passages", flush=True)

    uncertainty_mode = cli_args.mode or cfg.get('uncertainty_mode', 'mc_dropout')
    print(f"\n{'='*50}", flush=True)
    print(f"  GRASS MODE: {uncertainty_mode.upper()}", flush=True)
    print(f"{'='*50}\n", flush=True)

    if uncertainty_mode == 'ema':
        output_model_dir = train_with_ema_grass(
            stale_idx, stale_embs, c_id_to_idx, c_ids,
            corpus_lookup, qrels_dict, cfg, config, ctx, debug=cli_args.debug
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
