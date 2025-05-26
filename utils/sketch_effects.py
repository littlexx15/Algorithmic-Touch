# utils/sketch_effects.py

import cv2
import numpy as np
from PIL import Image

def to_sketch(
    img_pil: Image.Image,
    low_thresh: int = 50,
    high_thresh: int = 150,
    thick_k: int = 15,
    thin_k: int = 2,
    noise_sigma: float = 0.01,
    bin_thresh: float = 0.4,
    r_ratio: float = 1/3,
    sigma_ratio: float = 0.8
) -> Image.Image:
    """
    简笔画生成函数，返回 RGBA 白线图。
    """
    gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh).astype(np.float32) / 255.0

    h, w = edges.shape
    ys, xs = np.indices((h, w))
    cy, cx = h / 2.0, w / 2.0

    dist = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    d = dist / dist.max()

    r = int(min(h, w) * r_ratio)
    ksize = 2 * r + 1
    ksize = max(ksize, 3)
    sigma = max(r * sigma_ratio, 1.0)
    d = cv2.GaussianBlur(d.astype(np.float32), (ksize, ksize), sigmaX=sigma)

    big = cv2.dilate((edges * 255).astype(np.uint8), np.ones((thick_k, thick_k), np.uint8)).astype(np.float32) / 255.0
    small = cv2.dilate((edges * 255).astype(np.uint8), np.ones((thin_k, thin_k), np.uint8)).astype(np.float32) / 255.0

    w_s = 3 * d**2 - 2 * d**3
    final = big * (1 - w_s) + small * w_s
    final = np.clip(final + np.random.randn(h, w) * noise_sigma, 0, 1)

    mask = (final > bin_thresh).astype(np.uint8)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    ys2, xs2 = np.nonzero(mask)
    canvas[ys2, xs2] = [255, 255, 255, 255]

    return Image.fromarray(canvas, mode="RGBA")
