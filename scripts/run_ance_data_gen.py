"""
ANCE Inferencer — runs on GPU 1 in parallel with training.
Polls output_model_dir for new checkpoints → encode corpus+queries → FAISS → mine → write JSONL.
Signals training process via ann_dir/ready_{N} marker file.

Paper reference: Section 4 "Asynchronous Index Refresh", Figure 2, Appendix A.3
"""
import os
import sys
import time
import json
import pickle
import argparse
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'src'))

from utils.helpers import get_path, get_training_context, load_config, encode_to_pickle, build_faiss_index
import pandas as pd


def get_latest_ann_no(ann_dir: Path) -> int:
    nos = [int(f.name.split("_")[-1]) for f in ann_dir.glob("ready_*")
           if f.name.split("_")[-1].isdigit()]
    return max(nos) if nos else 0


def is_valid_checkpoint(ckpt_path: str) -> bool:
    """Inferencer only loads a checkpoint once optimizer.pt is present (fully written)."""
    p = Path(ckpt_path)
    return (p / "optimizer.pt").exists()


def _load_qrels(qrels_file):
    data = []
    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                data.append({'qid': parts[0], 'did': parts[2]})
    return pd.DataFrame(data).groupby('qid')['did'].apply(set).to_dict() if data else {}


def _load_corpus_lookup(corpus_file):
    lookup = {}
    with open(corpus_file) as f:
        for line in f:
            d = json.loads(line)
            lookup[d['docid']] = d['text']
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_model_dir', required=True)
    parser.add_argument('--ann_dir', required=True)
    parser.add_argument('--corpus_file', required=True)
    parser.add_argument('--query_file', required=True)
    parser.add_argument('--qrels_file', required=True)
    parser.add_argument('--recipe', default='ance')
    args = parser.parse_args()

    from transformers.trainer_utils import get_last_checkpoint

    ann_dir = Path(args.ann_dir)
    ann_dir.mkdir(exist_ok=True, parents=True)
    Path(args.output_model_dir).mkdir(exist_ok=True, parents=True)  # prevent FileNotFoundError before first checkpoint
    ctx = get_training_context(args.recipe)
    config = load_config()
    poll_interval = ctx['args'].get('data_gen_poll_interval', 60)
    n_negs = ctx['args']['train_group_size'] - 1
    mining_depth = ctx['args']['mining_depth']

    qrels_dict = _load_qrels(args.qrels_file)
    corpus_lookup = _load_corpus_lookup(args.corpus_file)

    last_checkpoint = None
    output_num = get_latest_ann_no(ann_dir) + 1
    print(f"[Inferencer] Polling {args.output_model_dir} every {poll_interval}s | "
          f"mining_depth={mining_depth}, n_negs={n_negs}", flush=True)

    # Run forever until parent process terminates this subprocess
    while True:
        next_checkpoint = get_last_checkpoint(str(args.output_model_dir))

        if (next_checkpoint is None or next_checkpoint == last_checkpoint
                or not is_valid_checkpoint(next_checkpoint)):
            time.sleep(poll_interval)
            continue

        print(f"[Inferencer] Checkpoint {Path(next_checkpoint).name} → "
              f"generating ANN data #{output_num}", flush=True)

        # Re-encode entire corpus with latest checkpoint (paper: "recomputes encodings of entire corpus")
        work_dir = ann_dir / f"work_{output_num}"
        work_dir.mkdir(exist_ok=True)
        encode_to_pickle(next_checkpoint, args.corpus_file, work_dir / "corpus.pkl", False, ctx, config)
        encode_to_pickle(next_checkpoint, args.query_file,  work_dir / "query.pkl",  True,  ctx, config)

        # Build FAISS IndexFlatIP and mine hard negatives
        # Paper Eq. 13: D^-_ANCE = ANN_{f(q,d)} \ D^+
        idx, _, c_ids = build_faiss_index(work_dir / "corpus.pkl")
        with open(work_dir / "query.pkl", 'rb') as f:
            q_data = pickle.load(f)
        _, indices = idx.search(q_data[0].astype(np.float32), mining_depth)

        mined_negs = {}
        for i, qid in enumerate([str(x) for x in q_data[1]]):
            pot = [c_ids[j] for j in indices[i] if j >= 0]
            true_negs = [d for d in pot if d not in qrels_dict.get(qid, set())]
            candidates = true_negs if true_negs else pot
            # Top-n hardest: earliest in FAISS-ranked list = highest similarity = hardest negative
            if len(candidates) >= n_negs:
                mined_negs[qid] = candidates[:n_negs]
            else:
                mined_negs[qid] = (candidates * (n_negs // max(len(candidates), 1) + 1))[:n_negs]

        # Write new JSONL training files to ann_dir/training_data_{N}/
        out_data_dir = ann_dir / f"training_data_{output_num}"
        out_data_dir.mkdir(exist_ok=True)
        for f_path in (get_path("processed") / ctx['args']['mixture_dir']).glob("*.jsonl"):
            if f_path.name.startswith('.'):
                continue
            with open(f_path) as f_in, open(out_data_dir / f_path.name, 'w') as f_out:
                for line in f_in:
                    d = json.loads(line)
                    if str(d['query_id']) in mined_negs:
                        d['negative_passages'] = [
                            {"docid": nid, "text": corpus_lookup.get(nid, "")}
                            for nid in mined_negs[str(d['query_id'])]
                        ]
                    f_out.write(json.dumps(d, ensure_ascii=False) + '\n')

        # Write ready marker AFTER all JSONL files are fully written (prevents partial reads)
        (ann_dir / f"ready_{output_num}").write_text(str(output_num))
        print(f"[Inferencer] ANN data #{output_num} ready. Waiting for next checkpoint...", flush=True)

        last_checkpoint = next_checkpoint
        output_num += 1


if __name__ == "__main__":
    main()
