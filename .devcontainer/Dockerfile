# Base image: CUDA 12.8 + cuDNN 9 + Ubuntu 22.04
# Supports sm_100 (B200), sm_90 (H100), sm_89 (RTX 4090/5090)
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# PyTorch 2.7 + CUDA 12.8 (supports B200 sm_100 out of the box)
RUN pip install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Core training dependencies — pinned to match working showui env
RUN pip install \
    transformers==4.47.0 \
    accelerate==0.34.2 \
    peft==0.18.1 \
    deepspeed==0.18.8 \
    bitsandbytes==0.49.2 \
    datasets==4.8.3 \
    tokenizers==0.21.4 \
    safetensors==0.7.0 \
    huggingface_hub==0.36.2

# ShowUI / Qwen2-VL specific
RUN pip install \
    qwen-vl-utils==0.0.14 \
    einops==0.8.2 \
    wandb==0.25.1 \
    tensorboard==2.20.0

# Vision / data utils
RUN pip install \
    pillow==12.0.0 \
    opencv-python==4.13.0.92 \
    scikit-image==0.25.2 \
    scikit-learn==1.7.2 \
    pycocotools==2.0.11 \
    numpy==2.2.6 \
    matplotlib==3.10.8 \
    Levenshtein==0.27.3 \
    nltk==3.9.3 \
    ultralytics==8.4.24

# Jupyter
RUN pip install \
    jupyterlab==4.5.6 \
    ipykernel==7.2.0

# Other utils
RUN pip install \
    openai==2.29.0 \
    selenium==4.12.0 \
    jax==0.6.2

# # Pre-compile DeepSpeed CUDA ops for B200
# RUN DS_BUILD_OPS=1 TORCH_CUDA_ARCH_LIST="10.0" \
#     pip install deepspeed==0.18.8 --force-reinstall

WORKDIR /workspace

# Verify install
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
