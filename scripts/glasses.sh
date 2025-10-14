#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

python sample_video.py \
    --dit-weight checkpoint/diffusion_pytorch_model.safetensors \
    --video-size 384 240 \
    --video-length 81 \
	--infer-steps 50 \
    --prompt "Add black glasses to the person's face." \
    --video "assets/glasses.mp4" \
    --seed 42 \
	--embedded-cfg-scale 1.0 \
    --cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./results
