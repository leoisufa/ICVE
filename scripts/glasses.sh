#!/bin/bash
# Single-GPU inference.
#   --video-length 61 -> latent frames = (61-1)/4 + 1 = 16
# Attention backend is auto (flash-attn if installed, else PyTorch SDPA).
# On an 80GB GPU the model fits without offload. For smaller GPUs, append
# `--use-cpu-offload` to trade speed for memory.

python sample_video.py \
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
    --save-path ./results
