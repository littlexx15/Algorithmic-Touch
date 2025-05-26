# cartoon_utils.py

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
    thin_k: int = 2,
    noise_sigma: float = 0.01,
    bin_thresh: float = 0.4,
    # —— 新增两个可调项 —— #
    r_ratio: float = 1/3,     # 控制过渡区域半径占最短边比例
    sigma_ratio: float = 0.8  # 控制高斯模糊强度，相对于 r
) -> Image.Image:
    """
    径向渐变简笔画（中心粗，外缘细 + 平滑过渡 + 随机发散）：
      - low_thresh/high_thresh: Canny 边缘检测阈值
      - thick_k:   中心粗线膨胀核大小
      - thin_k:    边缘细线膨胀核大小
      - noise_sigma: 随机噪声强度
      - bin_thresh: 二值化阈值（0–1）
      - r_ratio:   过渡区域半径占比（0–1），越小过渡越集中
      - sigma_ratio: 高斯模糊 sigma 相对于 r 的比例
    返回：RGBA 模式的 PIL.Image（白线＋透明底）
    """
    # 1. 灰度 & Canny 边缘检测
    gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh).astype(np.float32) / 255.0

    h, w = edges.shape
    ys, xs = np.indices((h, w))
    cy, cx = h / 2.0, w / 2.0

    # 2. 归一化径向距离 (0=center,1=corner)
    dist = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    d = dist / dist.max()

    # 3. 对 d 做一次自适应高斯平滑，让过渡更自然
    #    r_ratio 控制平滑影响范围，sigma_ratio 控制模糊强度
    r = int(min(h, w) * r_ratio)
    ksize = 2 * r + 1
    ksize = max(ksize, 3)  # 保证奇数且>=3
    sigma = max(r * sigma_ratio, 1.0)
    d = cv2.GaussianBlur(d.astype(np.float32), (ksize, ksize), sigmaX=sigma)

    # 4. 双尺度膨胀：big = 主轮廓，small = 细节
    big = cv2.dilate(
        (edges * 255).astype(np.uint8),
        np.ones((thick_k, thick_k), np.uint8)
    ).astype(np.float32) / 255.0
    small = cv2.dilate(
        (edges * 255).astype(np.uint8),
        np.ones((thin_k, thin_k), np.uint8)
    ).astype(np.float32) / 255.0

    # 5. smoothstep 插值：big→small
    w_s = 3 * d**2 - 2 * d**3
    final = big * (1 - w_s) + small * w_s

    # 6. 添加随机噪声
    final = np.clip(final + np.random.randn(h, w) * noise_sigma, 0, 1)

    # 7. 二值化 & 构造透明画布
    mask = (final > bin_thresh).astype(np.uint8)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    ys2, xs2 = np.nonzero(mask)
    canvas[ys2, xs2] = [255, 255, 255, 255]

    return Image.fromarray(canvas, mode="RGBA")
