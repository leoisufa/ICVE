#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

python sample_video.py \
    --dit-weight checkpoint/diffusion_pytorch_model.safetensors \
    --video-size 240 384 \
    --video-length 81 \
	--infer-steps 50 \
    --prompt "Remove the person on the left and fill the background with the desert scenery." \
    --video "assets/woman.mp4" \
    --seed 42 \
	--embedded-cfg-scale 1.0 \
    --cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./results
