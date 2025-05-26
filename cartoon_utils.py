# cartoon_utils.py

import torch
from PIL import Image
import cv2
import numpy as np

# —— PyTorch Hub 上的 AnimeGANv2 —— #
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
    输入：PIL.Image 任意大小
    输出：PIL.Image (512×512)
    """
    return face2paint(generator, img_pil)

def to_sketch(img_pil: Image.Image,
              low_threshold: int = 50,
              high_threshold: int = 150) -> Image.Image:
    """
    将 PIL 图像转成白色边缘线＋透明背景的简笔画效果。
    参数：
      img_pil: 原始 PIL.Image
      low_threshold/high_threshold: Canny 算子阈值
    返回：
      RGBA 模式的 PIL.Image，只有边缘为白，其余透明。
    """
    # 1. 转灰度
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    # 2. Canny 边缘检测
    edges = cv2.Canny(cv_img, low_threshold, high_threshold)
    # 3. 新建透明画布
    h, w = edges.shape
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    # 4. 边缘像素设为白（RGBA 都 255）
    ys, xs = np.nonzero(edges)
    canvas[ys, xs] = [255, 255, 255, 255]
    # 5. 返回 PIL
    return Image.fromarray(canvas, mode="RGBA")
