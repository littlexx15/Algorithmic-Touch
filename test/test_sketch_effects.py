from PIL import Image
from utils.sketch_effects import to_sketch

def test_to_sketch_empty_image():
    img = Image.new("RGB", (100, 100), color="white")
    # 只要不抛异常就算通过
    sketch = to_sketch(img, low_thresh=0, high_thresh=0,
                      thick_k=1, thin_k=1,
                      noise_sigma=0, bin_thresh=1.1,
                      r_ratio=0, sigma_ratio=0)
    assert sketch.mode == "RGBA"
    assert sketch.size == (100, 100)
