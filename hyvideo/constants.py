import os
import torch

__all__ = [
    "C_SCALE",
    "PROMPT_TEMPLATE",
    "MODEL_BASE",
    "PRECISIONS",
    "NORMALIZATION_TYPE",
    "ACTIVATION_TYPE",
    "VAE_PATH",
    "TEXT_ENCODER_PATH",
    "TOKENIZER_PATH",
    "TEXT_PROJECTION",
    "DATA_TYPE",
    "NEGATIVE_PROMPT",
]

PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

# =================== Constant Values =====================
# Computation scale factor, 1P = 1_000_000_000_000_000. Tensorboard will display the value in PetaFLOPS to avoid
# overflow error when tensorboard logging values.
C_SCALE = 1_000_000_000_000_000

# When using decoder-only models, we must provide a prompt template to instruct the text encoder
# on how to generate the text.
# --------------------------------------------------------------------
PROMPT_TEMPLATE_ENCODE = ("{}")
PROMPT_TEMPLATE_ENCODE_VIDEO = ("{}")

NEGATIVE_PROMPT = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"

PROMPT_TEMPLATE = {
    "dit-llm-encode": {
        "template": PROMPT_TEMPLATE_ENCODE,
        "crop_start": 0,
    },
    "dit-llm-encode-video": {
        "template": PROMPT_TEMPLATE_ENCODE_VIDEO,
        "crop_start": 0,
    },
}

# ======================= Model ======================
PRECISIONS = {"fp32", "fp16", "bf16"}
NORMALIZATION_TYPE = {"layer", "rms"}
ACTIVATION_TYPE = {"relu", "silu", "gelu", "gelu_tanh"}

# =================== Model Path =====================
MODEL_BASE = os.getenv("MODEL_BASE", "./ckpts")

# =================== Data =======================
DATA_TYPE = {"image", "video", "image_video"}

# 3D VAE
VAE_PATH = {"884-16c-hy": f"{MODEL_BASE}/hunyuan-video-t2v-720p/vae"}

# Text Encoder
TEXT_ENCODER_PATH = {
    "clipL": f"{MODEL_BASE}/text_encoder_2",
    "llm": f"{MODEL_BASE}/text_encoder",
}

# Tokenizer
TOKENIZER_PATH = {
    "clipL": f"{MODEL_BASE}/text_encoder_2",
    "llm": f"{MODEL_BASE}/text_encoder",
}

TEXT_PROJECTION = {
    "linear",  # Default, an nn.Linear() layer
    "single_refiner",  # Single TokenRefiner. Refer to LI-DiT
}
