<div align="center">
  <!-- Project Title -->
  <h1>
    ICVE: In-Context Learning with Unpaired Clips for Instruction-based Video Editing
  </h1>
  <!-- Project Badges -->
  <p>
    <a href="https://arxiv.org/abs/2510.14648">
      <img src="https://img.shields.io/badge/arXiv-2510.14648-b31b1b.svg" alt="arXiv"/>
    </a>
    <a href="https://huggingface.co/leoisufa/ICVE">
      <img src="https://img.shields.io/badge/HuggingFace-Model-yellow.svg" alt="HuggingFace"/>
    </a>
  </p>
</div>

<div align="center">
  <strong>Xinyao Liao<sup>1,2</sup></strong>,
  <strong>Xianfang Zeng<sup>2</sup></strong>,
  <strong>Ziye Song<sup>1</sup></strong>,
  <strong>Zhoujie Fu<sup>1,2</sup></strong>,
  <strong>Gang Yu<sup>2*</sup></strong>,
  <strong>Guosheng Lin<sup>1*</sup></strong>
  <br><br>
  <b>
    <sup>1</sup> Nanyang Technological University
    <a href="#">
      <img src="assets/images/Picture 1.png" alt="NTU Logo"
           style="margin-bottom: -4px; height: 22px;">
    </a>
      
    <sup>2</sup> StepFun
    <a href="#">
      <img src="assets/images/Picture 2.png" alt="StepFun Logo"
           style="margin-bottom: -4px; height: 22px;">
    </a>
  </b>
</div>

<div align="center">
  <img src="assets/gifs/output_small.gif" 
       alt="Demo GIF" 
       width="100%" 
       style="max-width:900px;">
</div>

**Star us if you find this project useful! ⭐**

### **The new model will be coming soon. Wan2.1-T2V-14B fondation model.**

## 🎉 Updates
- [12/2025] ⚠️ Fix the GPU OOM bug on 480P inference.
- [10/2025] 🔥 [Model checkpoints](https://huggingface.co/leoisufa/ICVE) is released!
- [10/2025] 🔥 [Codebase](https://github.com/leoisufa/ICVE) is relased!

## 🧩 Overview
ICVE proposes a low-cost pretraining strategy for instruction-based video editing via in-context learning from unpaired clips. Built upon [HunyuanVideoT2V](https://github.com/Tencent-Hunyuan/HunyuanVideo), it first learns editing concepts from about **1M** unpaired videos, then fine-tunes on **<150K** paired editing data for improved instruction alignment and visual quality — enabling general editing operations guided by natural language.

## 🎥 Video Demo
<p align="center">
  <a href="https://youtu.be/ZPXPMJUJnwU" target="_blank">
    <img src="https://img.youtube.com/vi/ZPXPMJUJnwU/maxresdefault.jpg" 
         alt="ICVE Demo Video" 
         width="80%" 
         style="max-width:900px; border-radius:10px; box-shadow:0 0 10px rgba(0,0,0,0.15);">
  </a>
  <br>
  <em>Click the image above to watch the full video on YouTube 🎬</em>
</p>

## 🛠️ Dependencies and Installation

Begin by cloning the repository:
```shell
git clone https://github.com/leoisufa/ICVE.git
cd ICVE
```

We recommend CUDA versions 12.4 or 11.8 for the manual installation.

```shell
# 1. Create conda environment
conda create -n icve python==3.10.9

# 2. Activate the environment
conda activate icve

# 3. Install PyTorch and other dependencies using conda
# For CUDA 11.8
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# For CUDA 12.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. Install pip dependencies
python -m pip install -r requirements.txt

# 5. Install flash attention v2 for acceleration (requires CUDA 11.8 or above)
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
```

## 🧱 Download Models
1. **HunyuanVideo Pretrained Weights**  
   Follow the official HunyuanVideo instructions here:  
   👉 [Download Pretrained Models](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/ckpts/README.md)  
   and place the downloaded weights into the `ckpts/` directory as shown above.
2. **ICVE Checkpoint**  
Download the our model weights from  
👉 [Hugging Face](https://huggingface.co/leoisufa/ICVE)  
and place them in the `checkpoint/` directory.

The folder structure of this project should look like this after setup:
```shell
ICVE/
├── assets/
├── checkpoint/ # Our model checkpoint
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── ckpts/  # Pretrained weights from HunyuanVideo
│   ├── hunyuan-video-t2v-720p
│   ├── text_encoder
│   └── text_encoder_2
├── hyvideo/
├── scripts/
├── requirements.txt
├── sample_video.py
└── README.md
``` 

## 🚀 Running the Demos

> **OOM Bug Fix:** We have fixed an OOM issue caused by the VAE not enabling `enable_tiling()` during original video encoding. After the fix, running inference on an original video with a resolution of 480×768×77 requires approximately 54 GB of GPU memory.

You can directly run the provided demo scripts under the [`scripts/`](./scripts) directory.
 
Alternatively, you can manually run the example command below:
```bash
python sample_video.py \
    --dit-weight checkpoint/diffusion_pytorch_model.safetensors \
    --video-size 768 480 \
    --video-length 77 \
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
```

## 🙏 Acknowledgements
We thank the following prior art for their excellent open source work: 
- [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [VACE](https://github.com/ali-vilab/VACE)

## 🔗 BibTeX
If you find [ICEV](https://arxiv.org/abs/2510.14648) useful for your research and applications, please cite using this BibTeX:
```BibTeX
@article{liao2025context,
  title={In-Context Learning with Unpaired Clips for Instruction-based Video Editing},
  author={Liao, Xinyao and Zeng, Xianfang and Song, Ziye and Fu, Zhoujie and Yu, Gang and Lin, Guosheng},
  journal={arXiv preprint arXiv:2510.14648},
  year={2025}
}
```