# cartoon_utils.py

import cv2
import numpy as np
from PIL import Image

def to_cartoon_fast(img_pil: Image.Image,
                    sigma_s: float = 60,
                    sigma_r: float = 0.07) -> Image.Image:
    """
    用 OpenCV 的 stylization 做卡通化/画风转换：
      - sigma_s 控制保留细节程度（0–200）
      - sigma_r 控制边缘保留程度（0–1）
    运行速度极快，纯 CPU 大约 50–100ms/张。
    """
    # 1. PIL -> OpenCV BGR
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    # 2. 卡通化
    cartoon_bgr = cv2.stylization(bgr, sigma_s=sigma_s, sigma_r=sigma_r)
    # 3. BGR -> PIL RGB
    cartoon_rgb = cv2.cvtColor(cartoon_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cartoon_rgb)
