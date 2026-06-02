#!/bin/bash
# Multi-GPU sequence-parallel (SP) inference.
#
# Usage:  bash scripts/glasses_multigpu.sh [NUM_GPUS]   (default 8)
#
# The SP degree equals the number of processes (WORLD_SIZE). The latent is
# sharded along the temporal axis across GPUs, so the latent frame count must
# be divisible by NUM_GPUS:
#       latent_T = (video_length - 1) / 4 + 1
#       video-length 61 -> latent_T 16  (divisible by 2/4/8)
# SP mode keeps the full model on each GPU (no CPU offload). Attention backend
# is auto (flash-attn if installed, else PyTorch SDPA).

NGPU=${1:-8}

torchrun --nproc_per_node=${NGPU} --master_port=29501 sample_video.py \
    --dit-weight checkpoint/diffusion_pytorch_model.safetensors \
    --video-size 768 480 \
    --video-length 61 \
    --infer-steps 50 \
    --prompt "Add black glasses to the person's face." \
    --video "assets/glasses.mp4" \
    --seed 42 \
    --embedded-cfg-scale 1.0 \
    --cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --attn-mode auto \
    --save-path ./results_multigpu
