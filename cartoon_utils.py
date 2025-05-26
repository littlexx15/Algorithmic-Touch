import cv2
import numpy as np
from PIL import Image
import torch

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
    """
    return face2paint(generator, img_pil)


def to_sketch(
    img_pil: Image.Image,
    low_thresh: int = 50,
    high_thresh: int = 150,
    thick_k: int = 15,
    thin_k: int = 1,
    noise_sigma: float = 0.005
) -> Image.Image:
    """
    径向渐变简笔画：
      - low_thresh/high_thresh: Canny 阈值
      - thick_k: 中心粗线膨胀核
      - thin_k: 边缘细线膨胀核
      - noise_sigma: 随机噪声强度
    """
    # 1. 灰度 + 边缘检测
    gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh).astype(np.float32) / 255.0

    h, w = edges.shape
    ys, xs = np.indices((h, w))
    cy, cx = h / 2.0, w / 2.0
    # 归一化距离 (0=center,1=corner)
    dist = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    d = dist / dist.max()

    # 2. 双尺度膨胀
    big   = cv2.dilate((edges*255).astype(np.uint8), np.ones((thick_k, thick_k), np.uint8))
    small = cv2.dilate((edges*255).astype(np.uint8), np.ones((thin_k, thin_k), np.uint8))
    big   = big.astype(np.float32)   / 255.0
    small = small.astype(np.float32) / 255.0

    # 3. 径向插值：中心粗，外缘细
    final = big * (1 - d) + small * d

    # 4. 加随机噪声
    final = np.clip(final + np.random.randn(h, w) * noise_sigma, 0, 1)

    # 5. 二值化 & 透明画布
    mask = (final > 0.2).astype(np.uint8)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    ys2, xs2 = np.nonzero(mask)
    canvas[ys2, xs2] = [255, 255, 255, 255]

    return Image.fromarray(canvas, mode="RGBA")
