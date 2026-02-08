#!/usr/bin/env bash
# Setup script to install packages in the Singularity container
# Run this on DelftBlue: bash scripts/setup_container_packages.sh

CONTAINER="/scratch/${USER}/containers/pytorch_2.1.sif"

echo "Installing Python packages in container..."

# Install core packages
singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    pip install --user --no-cache-dir \
        numpy==1.24.4 \
        pandas>=2.0.0 \
        pyyaml>=6.0 \
        tqdm>=4.65.0 \
        spacy>=3.5.0 \
        importlib_metadata \
        transformers==4.36.0 \
        sentence-transformers==2.2.2 \
        datasets==2.14.0 \
        accelerate==0.25.0 \
        safetensors==0.4.0 \
        peft==0.7.0 \
        deepspeed==0.12.0 \
        faiss-gpu==1.7.4 \
        pyserini==0.22.0 \
        pytrec-eval

echo "Installing GradCache from GitHub..."
singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    pip install --user git+https://github.com/luyug/GradCache.git

echo "Installing Tevatron from stable commit (before multimodal code)..."
# Use commit from before Qwen multimodal integration
singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    pip install --user "git+https://github.com/texttron/tevatron.git@v0.1.0"

echo "✅ Package installation complete!"
echo ""
echo "Verifying installation..."
singularity exec --nv \
    --bind /scratch/${USER}:/scratch/${USER} \
    --bind /home/${USER}:/home/${USER} \
    ${CONTAINER} \
    python3 -c "
import torch
import transformers
from tevatron.retriever.modeling import DenseModel
from grad_cache import GradCache
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print('✅ All imports successful!')
"
