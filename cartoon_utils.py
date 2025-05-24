# cartoon_utils.py

import torch
from PIL import Image

# —— PyTorch Hub 上的 AnimeGANv2 —— #
# 使用 AK391/animegan2-pytorch:main
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载生成器
generator = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "generator",
    pretrained="face_paint_512_v2",
    device=DEVICE
)
# 加载封装好的 face2paint 接口（直接接 PIL.Image → PIL.Image）
face2paint = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "face2paint",
    size=512,
    device=DEVICE,
    side_by_side=False
)

def to_animegan2(img_pil: Image.Image) -> Image.Image:
    """
    一键 AnimeGANv2 二次元动漫化。
    输入：PIL.Image 任意大小
    输出：PIL.Image (512×512)
    """
    # 生成并返回 PIL.Image
    return face2paint(generator, img_pil)
