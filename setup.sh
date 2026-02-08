#!/usr/bin/env bash
set -e  # Exit on any error

echo "=========================================="
echo "🚀 Dense Retrieval SOTA Setup Script"
echo "=========================================="
echo ""

# --- Step 1: Check Singularity/Apptainer ---
echo "📦 Checking for Singularity/Apptainer..."
if command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
    echo "✅ Found Singularity: $(singularity --version)"
elif command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
    echo "✅ Found Apptainer: $(apptainer --version)"
else
    echo "❌ Error: Neither Singularity nor Apptainer found!"
    echo "Please install one of them first:"
    echo "  - On DelftBlue: module load 2023r1 apptainer"
    exit 1
fi
echo ""

# --- Step 2: Setup directories ---
echo "📁 Creating directory structure..."
export DATA_BASE_DIR="/scratch/${USER}/dense-retrieval-SOTA"
export CONTAINER_DIR="/scratch/${USER}/containers"

mkdir -p logs
mkdir -p models
mkdir -p data/bright
mkdir -p data/processed
mkdir -p "${CONTAINER_DIR}"
mkdir -p "${DATA_BASE_DIR}"
echo "✅ Directories created"
echo ""

# --- Step 3: Container setup ---
CONTAINER="${CONTAINER_DIR}/pytorch_2.1.sif"
echo "🐳 Checking for PyTorch container..."

if [ -f "${CONTAINER}" ]; then
    echo "✅ Container found at: ${CONTAINER}"
else
    echo "⚠️  Container not found at: ${CONTAINER}"
    echo ""
    echo "📥 Attempting to pull PyTorch 2.1 container from Docker Hub..."
    echo "This may take 10-15 minutes..."

    # Try to pull from Docker Hub
    if ${CONTAINER_CMD} pull --dir "${CONTAINER_DIR}" docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime; then
        # Rename to expected name
        mv "${CONTAINER_DIR}/pytorch_pytorch_2.1.0-cuda11.8-cudnn8-runtime.sif" "${CONTAINER}" 2>/dev/null || true
        echo "✅ Container downloaded successfully"
    else
        echo "❌ Failed to download container automatically"
        echo ""
        echo "Please manually download the container:"
        echo "  singularity pull --dir ${CONTAINER_DIR} docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
        echo "  mv ${CONTAINER_DIR}/pytorch_pytorch_2.1.0-cuda11.8-cudnn8-runtime.sif ${CONTAINER}"
        echo ""
        echo "Or if you have the container elsewhere, copy/symlink it to:"
        echo "  ${CONTAINER}"
        exit 1
    fi
fi
echo ""

# --- Step 4: Install Tevatron ---
echo "📚 Installing Tevatron..."
export PYTHONPATH="${HOME}/dense-retrieval-SOTA/src:${PYTHONPATH}"

# Check if already installed
if ${CONTAINER_CMD} exec "${CONTAINER}" python -c "import tevatron" 2>/dev/null; then
    echo "⚠️  Tevatron already installed, reinstalling to ensure correct version..."
fi

${CONTAINER_CMD} exec "${CONTAINER}" \
    pip install --user --force-reinstall git+https://github.com/texttron/tevatron.git@8f31cd8

if ${CONTAINER_CMD} exec "${CONTAINER}" python -c "import tevatron" 2>/dev/null; then
    echo "✅ Tevatron installed successfully"
else
    echo "❌ Tevatron installation failed"
    exit 1
fi
echo ""

# --- Step 5: Apply Tevatron patches ---
echo "🔧 Applying Tevatron patches..."

TEVATRON_BASE="${HOME}/.local/lib/python3.10/site-packages/tevatron"

# Patch 1: Comment out Qwen import in dense.py (line 3)
echo "  → Patch 1/4: Removing Qwen import from dense.py..."
DENSE_FILE="${TEVATRON_BASE}/retriever/modeling/dense.py"
if [ -f "${DENSE_FILE}" ]; then
    sed -i.bak '3s/^from transformers import Qwen2_5OmniThinkerForConditionalGeneration/# from transformers import Qwen2_5OmniThinkerForConditionalGeneration/' "${DENSE_FILE}"
    echo "    ✅ Line 3 patched"
else
    echo "    ⚠️  File not found: ${DENSE_FILE}"
fi

# Patch 2: Comment out Qwen assignment in dense.py (line 43)
echo "  → Patch 2/4: Removing Qwen class assignment from dense.py..."
if [ -f "${DENSE_FILE}" ]; then
    sed -i.bak '43s/^TRANSFORMER_CLS = Qwen2_5OmniThinkerForConditionalGeneration/# TRANSFORMER_CLS = Qwen2_5OmniThinkerForConditionalGeneration/' "${DENSE_FILE}"
    echo "    ✅ Line 43 patched"
fi

# Patch 3: Remove MultiModalDenseModel from __init__.py
echo "  → Patch 3/4: Removing MultiModalDenseModel export from __init__.py..."
INIT_FILE="${TEVATRON_BASE}/retriever/modeling/__init__.py"
if [ -f "${INIT_FILE}" ]; then
    sed -i.bak 's/from \.dense import DenseModel, MultiModalDenseModel/from .dense import DenseModel/' "${INIT_FILE}"
    echo "    ✅ __init__.py patched"
else
    echo "    ⚠️  File not found: ${INIT_FILE}"
fi

# Patch 4: Add torch import to train.py
echo "  → Patch 4/4: Adding torch import to train.py..."
TRAIN_FILE="${TEVATRON_BASE}/retriever/driver/train.py"
if [ -f "${TRAIN_FILE}" ]; then
    # Check if torch is already imported
    if grep -q "^import torch" "${TRAIN_FILE}"; then
        echo "    ℹ️  torch already imported"
    else
        # Add import torch at the beginning after shebang/comments
        sed -i.bak '1i\
import torch
' "${TRAIN_FILE}"
        echo "    ✅ torch import added"
    fi
else
    echo "    ⚠️  File not found: ${TRAIN_FILE}"
fi

echo "✅ All patches applied successfully"
echo ""

# --- Step 6: Verify patches ---
echo "🔍 Verifying patches..."
PATCH_OK=true

# Check that Qwen import is commented
if grep -q "^# from transformers import Qwen2_5OmniThinkerForConditionalGeneration" "${DENSE_FILE}"; then
    echo "  ✅ Qwen import commented"
else
    echo "  ❌ Qwen import not properly commented"
    PATCH_OK=false
fi

# Check that torch is imported
if grep -q "^import torch" "${TRAIN_FILE}"; then
    echo "  ✅ torch imported in train.py"
else
    echo "  ❌ torch not imported in train.py"
    PATCH_OK=false
fi

if [ "$PATCH_OK" = true ]; then
    echo "✅ All patches verified"
else
    echo "⚠️  Some patches may not have applied correctly"
    echo "Please check the files manually"
fi
echo ""

# --- Step 7: Environment setup instructions ---
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Next steps:"
echo ""
echo "1️⃣  Download models and data for offline mode:"
echo "   singularity exec --nv \\"
echo "       --bind /scratch/\${USER}:/scratch/\${USER} \\"
echo "       --bind /home/\${USER}:/home/\${USER} \\"
echo "       ${CONTAINER} \\"
echo "       python scripts/prepare_models.py"
echo ""
echo "2️⃣  Preprocess the data:"
echo "   singularity exec --nv \\"
echo "       --bind /scratch/\${USER}:/scratch/\${USER} \\"
echo "       --bind /home/\${USER}:/home/\${USER} \\"
echo "       ${CONTAINER} \\"
echo "       python scripts/preprocessor.py"
echo ""
echo "3️⃣  Submit training jobs:"
echo "   sbatch scripts/run_inbatch_singularity.sh      # In-batch baseline"
echo "   sbatch scripts/run_crossbatch_singularity.sh   # Cross-batch (2048)"
echo "   sbatch scripts/run_ance_singularity.sh         # ANCE iterative"
echo ""
echo "4️⃣  Evaluate trained models:"
echo "   sbatch scripts/run_evaluate_singularity.sh"
echo ""
echo "📚 For detailed setup info, see DELFTBLUE_SETUP.md"
echo "=========================================="