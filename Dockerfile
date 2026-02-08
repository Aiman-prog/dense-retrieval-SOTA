# Dockerfile for Dense Retrieval with GradCache and Tevatron
# Based on NVIDIA NGC PyTorch container with CUDA 11.8

FROM nvcr.io/nvidia/pytorch:23.10-py3

# This image includes:
# - PyTorch 2.1.0 (compatible with GradCache)
# - CUDA 12.2 (backward compatible with 11.8)
# - cuDNN, NCCL, and other NVIDIA libraries

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (with version constraints to prevent PyTorch upgrade)
RUN pip install --no-cache-dir \
    numpy>=1.24.0,<1.26.0 \
    pandas>=2.0.0 \
    pyyaml>=6.0 \
    tqdm>=4.65.0 \
    spacy>=3.5.0 \
    importlib_metadata \
    transformers>=4.36.0,<4.50.0 \
    sentence-transformers>=2.2.0,<3.0.0 \
    datasets>=2.14.0,<3.0.0 \
    accelerate>=0.25.0,<0.35.0 \
    safetensors>=0.4.0,<0.5.0 \
    peft>=0.7.0,<0.12.0 \
    deepspeed>=0.12.0,<0.15.0 \
    faiss-gpu>=1.7.0,<1.9.0 \
    pyserini>=0.22.0,<0.30.0 \
    pytrec-eval \
    qwen-omni-utils

# Install GradCache from GitHub
RUN git clone https://github.com/luyug/GradCache.git /workspace/GradCache && \
    cd /workspace/GradCache && \
    pip install -e .

# Install Tevatron from GitHub
RUN git clone https://github.com/texttron/tevatron.git /workspace/tevatron && \
    cd /workspace/tevatron && \
    pip install -e .

# Set Python path
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Default command
CMD ["/bin/bash"]
