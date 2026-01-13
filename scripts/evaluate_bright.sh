#!/bin/bash
# Evaluate a trained model on BRIGHT dataset using Tevatron CLI commands
# USAGE: bash scripts/evaluate_bright.sh <MODEL_PATH> <DOMAIN>
# Example: bash scripts/evaluate_bright.sh /scratch/aimanabdulwaha/dense-retrieval-SOTA/models/inbatch_reasonir biology

set -x  # Debug mode

echo "=========================================="
echo "evaluate_bright.sh STARTED"
echo "=========================================="
echo "Arguments received:"
echo "  MODEL_PATH: $1"
echo "  DOMAIN: $2"
echo "  BATCH_SIZE (env): ${BATCH_SIZE}"
echo "  K (env): ${K}"
echo "=========================================="

MODEL_PATH=$1
DOMAIN=${2:-"biology"}
BATCH_SIZE=${BATCH_SIZE:-512}
TOP_K=${K:-10}

# DelftBlue paths
SCRATCH_DIR="/scratch/${USER}/dense-retrieval-SOTA"
HOME_DIR="/home/${USER}/dense-retrieval-SOTA"

# Prefer scratch
if [ -d "${SCRATCH_DIR}/data/processed/eval/${DOMAIN}" ] && [ -f "${SCRATCH_DIR}/data/processed/eval/${DOMAIN}/corpus.jsonl" ]; then
    PROCESSED_DIR="${SCRATCH_DIR}/data/processed/eval/${DOMAIN}"
    EVAL_OUTPUT_DIR="${SCRATCH_DIR}/data/evaluation/${DOMAIN}"
elif [ -d "${HOME_DIR}/data/processed/eval/${DOMAIN}" ] && [ -f "${HOME_DIR}/data/processed/eval/${DOMAIN}/corpus.jsonl" ]; then
    PROCESSED_DIR="${HOME_DIR}/data/processed/eval/${DOMAIN}"
    EVAL_OUTPUT_DIR="${HOME_DIR}/data/evaluation/${DOMAIN}"
else
    PROCESSED_DIR="${SCRATCH_DIR}/data/processed/eval/${DOMAIN}"
    EVAL_OUTPUT_DIR="${SCRATCH_DIR}/data/evaluation/${DOMAIN}"
fi

mkdir -p "${PROCESSED_DIR}"
mkdir -p "${EVAL_OUTPUT_DIR}"

CODE_DIR="${SCRATCH_DIR}"
if [ ! -d "${CODE_DIR}/src" ]; then
    CODE_DIR="/home/${USER}/dense-retrieval-SOTA"
fi
CONFIG_PATH="${SCRATCH_DIR}/config/config.yaml"
if [ ! -f "${CONFIG_PATH}" ]; then
    CONFIG_PATH="/home/${USER}/dense-retrieval-SOTA/config/config.yaml"
fi

# 1. Prepare Data
if [ ! -f "${PROCESSED_DIR}/corpus.jsonl" ] || [ ! -f "${PROCESSED_DIR}/queries.jsonl" ] || [ ! -f "${PROCESSED_DIR}/qrels.txt" ]; then
    echo "Step 1: Preprocessing data for $DOMAIN..."
    python -c "
from pathlib import Path
import sys
project_root = Path('${CODE_DIR}')
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
from src.data.bright_loader import BRIGHTLoader
from src.data.preprocessor import BRIGHTPreprocessor
from src.utils.helpers import load_config

config = load_config('${CONFIG_PATH}')
loader = BRIGHTLoader(config_path='${CONFIG_PATH}')
loader.load_dataset()

preprocessor = BRIGHTPreprocessor(output_dir='${PROCESSED_DIR}')
corpus_df = loader.get_corpus('${DOMAIN}')
queries_df = loader.get_queries('${DOMAIN}')
qrels_df = loader.get_qrels('${DOMAIN}')

# These functions now add both text_id and docid/query_id
preprocessor.prepare_tevatron_corpus(corpus_df, 'corpus.jsonl')
preprocessor.prepare_tevatron_queries(queries_df, 'queries.jsonl')
preprocessor.prepare_trec_qrels(qrels_df, 'qrels.txt')
print('✅ Data prepared')
" || { echo "ERROR: Data preprocessing failed!"; exit 1; }
else
    echo "DEBUG: Preprocessing files already exist."
fi

# 2. Encode Corpus
echo "Step 2: Encoding Corpus..."
CORPUS_FILE="${PROCESSED_DIR}/corpus.jsonl"
python -m tevatron.driver.encode \
    --output_dir="${EVAL_OUTPUT_DIR}/corpus_emb" \
    --model_name_or_path="${MODEL_PATH}" \
    --encode_in_path="${CORPUS_FILE}" \
    --encoded_save_path="${EVAL_OUTPUT_DIR}/corpus.pkl" \
    --fp16 \
    --per_device_eval_batch_size="${BATCH_SIZE}" || exit 1

# 3. Encode Queries
echo "Step 3: Encoding Queries..."
QUERIES_FILE="${PROCESSED_DIR}/queries.jsonl"
python -m tevatron.driver.encode \
    --output_dir="${EVAL_OUTPUT_DIR}/query_emb" \
    --model_name_or_path="${MODEL_PATH}" \
    --encode_in_path="${QUERIES_FILE}" \
    --encoded_save_path="${EVAL_OUTPUT_DIR}/query.pkl" \
    --fp16 \
    --per_device_eval_batch_size="${BATCH_SIZE}" || exit 1

# 4. Retrieval
echo "Step 4: Retrieving Top-${TOP_K}..."
python -m tevatron.faiss_retriever \
    --query_reps "${EVAL_OUTPUT_DIR}/query.pkl" \
    --passage_reps "${EVAL_OUTPUT_DIR}/corpus.pkl" \
    --depth "${TOP_K}" \
    --batch_size "${BATCH_SIZE}" \
    --save_ranking_to "${EVAL_OUTPUT_DIR}/ranking.txt" || exit 1

# 5. Scoring (TREC Eval)
echo "Step 5: Scoring..."
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"

# --- CHANGE IS HERE ---
# We pass corpus_file and queries_file so score.py can map IDs
python -m src.evaluation.score \
    --qrels "${PROCESSED_DIR}/qrels.txt" \
    --ranking "${EVAL_OUTPUT_DIR}/ranking.txt" \
    --domain "${DOMAIN}" \
    --corpus_file "${PROCESSED_DIR}/corpus.jsonl" \
    --queries_file "${PROCESSED_DIR}/queries.jsonl"
# ----------------------

echo "========================================================"
echo "Evaluation completed!"
echo "========================================================"
