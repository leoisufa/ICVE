# **ICVE: In-Context Learning with Unpaired Clips for Instruction-based Video Editing**
*Arxiv 2025*

**Xinyao Liao**<sup>1,2</sup>, **Xianfang Zeng**<sup>2</sup>, **Ziye Song**<sup>1</sup>, **Zhoujie Fu**<sup>1,2</sup>, **Gang Yu**<sup>2*</sup>, **Guosheng Lin**<sup>1*</sup>  
<sup>1</sup> Nanyang Technological University  <sup>2</sup> StepFun

**Project Leader:** *Xianfang Zeng*  
**Corresponding Authors:** *Gang Yu, Guosheng Lin*

## 🧩 Overview
ICVE proposes a **low-cost pretraining strategy** for instruction-based video editing via **in-context learning from unpaired clips**. Built upon [**HunyuanVideoT2V**](https://github.com/Tencent-Hunyuan/HunyuanVideo), it first learns editing concepts from **about 1M unpaired videos**, then fine-tunes on **<150K paired editing data** for improved instruction alignment and visual quality — enabling general editing operations guided by natural language.

## 🎥 Demo
<p align="center">
  <a href="https://youtu.be/fmjmOWqQo88">
    <img src="https://img.youtube.com/vi/fmjmOWqQo88/0.jpg" 
         alt="ICVE Demo Video" 
         width="80%" 
         style="max-width:900px;">
  </a>
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

## 🧱 Download Pretrained Models
1. **HunyuanVideo Pretrained Weights**  
   Follow the official HunyuanVideo instructions here:  
   👉 [Download Pretrained Models](https://github.com/Tencent-Hunyuan/HunyuanVideo?tab=readme-ov-file#-download-pretrained-models)  
   and place the downloaded weights into the `ckpts/` directory as shown above.
2. **ICVE Checkpoint**  
   Download the our model weights from  
   👉 [Hugging Face](https://huggingface.co)  
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
You can directly run the provided demo scripts under the [`scripts/`](./scripts) directory.
 
Alternatively, you can manually run the example command below:
```bash
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
```

## 🔗 BibTeX
If you find [ICEV]() useful for your research and applications, please cite using this BibTeX:
```BibTeX
@article{liao2025icve,
  title   = {In-Context Learning with Unpaired Clips for Instruction-based Video Editing},
  author  = {Liao, Xinyao and Zeng, Xianfang and Song, Ziye and Fu, Zhoujie and Yu, Gang and Lin, Guosheng},
  journal = {arXiv preprint arXiv:2503.XXXX},
  year    = {2025}
}
```



## 🙏 Acknowledgements
This work builds upon the open-source efforts of [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) and [FastVideo](https://github.com/hao-ai-lab/FastVideo).