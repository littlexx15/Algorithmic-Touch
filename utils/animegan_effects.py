# utils/animegan_effects.py

import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "generator",
    pretrained="face_paint_512_v2",
    device=DEVICE
)
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
    """
    return face2paint(generator, img_pil)
