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
    # NOTE: the live DelftBlue env does not run this container's torch — ~/.local
    # provides torch 2.10.0+cu128 and shadows it. See P7 / docs/DELFTBLUE_ENVIRONMENT.md.
    echo "📥 Attempting to pull PyTorch 2.1 container from Docker Hub..."
    echo "This may take 10-15 minutes..."

    # Try to pull from Docker Hub (pull directly to target path)
    if ${CONTAINER_CMD} pull "${CONTAINER}" docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime; then
        echo "✅ Container downloaded successfully"
    else
        echo "❌ Failed to download container automatically"
        echo ""
        echo "Please manually download the container:"
        echo "  ${CONTAINER_CMD} pull ${CONTAINER} docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
        echo ""
        echo "Or if you have the container elsewhere, copy/symlink it to:"
        echo "  ${CONTAINER}"
        exit 1
    fi
fi

# Final container verification
if [ ! -f "${CONTAINER}" ]; then
    echo "❌ Error: Container not found at ${CONTAINER} after setup"
    echo "Please download it manually:"
    echo "  ${CONTAINER_CMD} pull ${CONTAINER} docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
    exit 1
fi
echo ""

# --- Step 4: Install Tevatron ---
echo "📚 Installing Tevatron..."
export PYTHONPATH="${HOME}/dense-retrieval-SOTA/src:${PYTHONPATH}"

# Install dependencies from requirements-hpc.txt.
# The pins were chosen for the container's PyTorch 2.1, but the live environment runs
# torch 2.10.0+cu128 from ~/.local. The pins still work; do not 'fix' them upward (P7).
${CONTAINER_CMD} exec "${CONTAINER}" \
    pip install --user -r requirements-hpc.txt

# Install tevatron and GradCache from tarballs (no git required in container)
${CONTAINER_CMD} exec "${CONTAINER}" \
    pip install --user --no-deps https://github.com/texttron/tevatron/archive/8f31cd8.tar.gz

${CONTAINER_CMD} exec "${CONTAINER}" \
    pip install --user --no-deps https://github.com/luyug/GradCache/archive/main.tar.gz

if ${CONTAINER_CMD} exec "${CONTAINER}" python -c "import tevatron" 2>/dev/null; then
    echo "✅ Tevatron installed successfully"
else
    echo "❌ Tevatron installation failed"
    exit 1
fi
echo ""

# --- Step 5: Apply Tevatron patches ---
echo "🔧 Applying Tevatron patches..."

# Define the location of the installed Tevatron package
TEVATRON_BASE="${HOME}/.local/lib/python3.10/site-packages/tevatron"

# Patch 1: Run the external patch script
echo "  → Patch 1/3: Running patch_tevatron.py script..."
if [ -f "scripts/patch_tevatron.py" ]; then
    ${CONTAINER_CMD} exec "${CONTAINER}" python3 scripts/patch_tevatron.py "${TEVATRON_BASE}"
else
    echo "  ❌ Error: scripts/patch_tevatron.py not found!"
    exit 1
fi

# Patch 2: Add torch import to train.py
echo "  → Patch 2/3: Adding torch import to train.py..."
TRAIN_FILE="${TEVATRON_BASE}/retriever/driver/train.py"
if [ -f "${TRAIN_FILE}" ]; then
    if grep -q "^import torch" "${TRAIN_FILE}"; then
        echo "    ℹ️  torch already imported"
    else
        sed -i.bak '1i\
import torch
' "${TRAIN_FILE}"
        echo "    ✅ torch import added"
    fi
else
    echo "    ⚠️  File not found: ${TRAIN_FILE}"
fi

echo ""

# --- Step 6: Verify patches ---
echo "  → Patch 3/3: Verifying all patches..."
PATCH_OK=true

# Check that NO active Qwen/multimodal references remain
REMAINING=$(grep -r --include="*.py" -l "qwen_omni_utils\|Qwen2_5Omni\|MultiModalDenseModel\|encoder\.visual" "${TEVATRON_BASE}" 2>/dev/null | xargs grep -v "^#" 2>/dev/null | grep -c "qwen_omni_utils\|Qwen2_5Omni\|MultiModalDenseModel\|encoder\.visual" 2>/dev/null || echo 0)
if [ "$REMAINING" -gt 0 ]; then
    echo "  ❌ ${REMAINING} uncommented Qwen/multimodal references still found"
    PATCH_OK=false
else
    echo "  ✅ All Qwen/multimodal references removed"
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
echo "       python src/data/preprocessor.py"
echo ""
echo "3️⃣  Submit training jobs:"
echo "   sbatch scripts/run_inbatch_singularity.sh      # In-batch baseline"
echo "   sbatch scripts/run_crossbatch_singularity.sh   # Cross-batch (2048)"
echo "   sbatch scripts/run_ance_singularity.sh         # ANCE iterative"
echo ""
echo "4️⃣  Evaluate trained models:"
echo "   sbatch scripts/run_evaluate_singularity.sh"
echo ""
echo "📚 For detailed setup info, see docs/DELFTBLUE_SETUP.md"
echo "=========================================="