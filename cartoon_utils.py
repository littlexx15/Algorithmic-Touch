# cartoon_utils.py

import torch
from PIL import Image

# 1. 选择设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 从 PyTorch Hub 加载 AnimeGANv2 生成器（v2 风格）
#    'face_paint_512_v2' 对应增强的鲁棒性和风格化
model = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "generator",
    pretrained="face_paint_512_v2",
    device=DEVICE
)
# 3. 加载“上色”接口，size=512，保持原图结构
face2paint = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "face2paint",
    size=512,
    device=DEVICE,
    side_by_side=False
)

def to_animegan2(img_pil: Image.Image) -> Image.Image:
    """
    一键式 AnimeGANv2 二次元动漫化。
    输入：PIL.Image
    输出：PIL.Image（512×512）
    """
    # 直接调用 face2paint，返回 PIL.Image
    return face2paint(model, img_pil)
