from PIL import Image
from utils.image_helpers import crop_and_resize

def test_crop_and_resize_various_ratios():
    # 宽很大，高很小
    img1 = Image.new("RGB", (800, 200))
    out1 = crop_and_resize(img1, target_ratio=3/2, width=300)
    assert out1.size == (300, 200)  # 3:2 → 高度 200

    # 宽很小，高很大
    img2 = Image.new("RGB", (200, 800))
    out2 = crop_and_resize(img2, target_ratio=3/2, width=300)
    assert out2.size == (300, 200)
