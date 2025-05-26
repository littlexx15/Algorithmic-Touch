# cartoon_utils.py (新增或替换 to_sketch)

import cv2
import numpy as np
from PIL import Image
import random

def to_sketch(
    img_pil: Image.Image,
    low_threshold: int = 50,
    high_threshold: int = 150,
    thickness: int = 1,
    random_thickness: bool = False,
    max_thickness: int = 5,
    center_fade: bool = False,
    fade_radius: float = 0.5
) -> Image.Image:
    """
    生成白色边缘＋透明背景的简笔画。
    参数：
      low_threshold, high_threshold: Canny 边缘检测阈值
      thickness: 膨胀核大小（固定模式时）；实际核为 (thickness,thickness)
      random_thickness: 是否随机粗细
      max_thickness: 随机模式的最大膨胀核大小
      center_fade: 是否做中粗外细效果
      fade_radius: 中心半径比例 (0-1)，越大中心范围越大
    返回：
      RGBA 模式的 PIL 图像
    """
    # 1. 转灰度
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    # 2. Canny 边缘
    edges = cv2.Canny(cv_img, low_threshold, high_threshold)

    # 3. 决定膨胀核大小
    if random_thickness:
        k = random.randint(1, max_thickness)
    else:
        k = max(1, thickness)
    kernel = np.ones((k, k), np.uint8)
    edges_thick = cv2.dilate(edges, kernel, iterations=1)

    # 4. 如果不做中心渐变，直接用 edges_thick，否则做混合
    if not center_fade:
        final_edges = edges_thick
    else:
        # 4a. 保留“细”版本
        edges_thin = edges.copy()
        h, w = edges.shape
        # 4b. 计算归一化距离图 dist_norm：中心 (0,0) 到角落 (1,1)
        yy, xx = np.indices((h, w))
        cx, cy = w/2, h/2
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        dist_norm = dist / dist.max()  # 0=center, 1=corner

        # 4c. 中心半径比例
        mask_center = dist_norm <= fade_radius
        # 4d. 合成：中心用厚边，其余用细边
        final_edges = np.where(mask_center, edges_thick, edges_thin)

    # 5. 构造 RGBA 画布
    h, w = final_edges.shape
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    # 6. 非零像素设为白（包括 alpha）
    ys, xs = np.nonzero(final_edges)
    canvas[ys, xs] = [255, 255, 255, 255]

    return Image.fromarray(canvas, mode="RGBA")
