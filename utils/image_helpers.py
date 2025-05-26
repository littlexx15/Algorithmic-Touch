# utils/image_helpers.py

import io
import base64
from PIL import Image

def pil_to_dataurl(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def crop_and_resize(img: Image.Image,
                    target_ratio: float = 3/2,
                    width: int = 600) -> Image.Image:
    w, h = img.size
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    new_h = int(width / target_ratio)
    return img.resize((width, new_h), Image.LANCZOS)
