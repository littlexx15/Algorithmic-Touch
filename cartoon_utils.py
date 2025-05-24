# cartoon_utils.py

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

MODEL_ID = "runwayml/stable-diffusion-v1-5"

# 1. 选择设备和半精度
if torch.cuda.is_available():
    device, torch_dtype = "cuda", torch.float16
elif torch.backends.mps.is_available():
    device, torch_dtype = "mps", None
else:
    device, torch_dtype = "cpu", None

# 2. 加载管道（只有 CUDA 时才用半精度）
load_kwargs = {"torch_dtype": torch_dtype} if torch_dtype else {}
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, **load_kwargs)
pipe = pipe.to(device)

# 3. 开启 slicing 加速
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass  # 没装 xformers 就跳过

def to_cartoon_sd(
    img: Image.Image,
    prompt: str = "cute cartoon pastel illustration, flat style, preserve detail",
    strength: float = 0.25,
    guidance_scale: float = 8.0,
    steps: int = 20
) -> Image.Image:
    """
    极致加速的 Img2Img 卡通化：
    - 分辨率 256×256
    - num_inference_steps = 20
    - attention & VAE slicing
    """
    init_img = img.convert("RGB").resize((256, 256))
    out = pipe(
        prompt=prompt,
        image=init_img,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps
    ).images[0]
    return out
