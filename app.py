import io
import base64
from PIL import Image, ImageOps
from flask import Flask, render_template, request

from models.predict import predict
from utils.sketch_effects import to_sketch
from utils.disease_info import disease_info

app = Flask(__name__)

def pil_to_dataurl(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def crop_and_resize(img: Image.Image,
                    target_ratio: float = 3/2,
                    width: int = 600) -> Image.Image:
    """
    将 img 裁剪到 target_ratio（宽/高），再缩放到给定宽度。
    """
    w, h = img.size
    current_ratio = w / h

    # 裁剪中心区域
    if current_ratio > target_ratio:
        # 太宽：左右裁
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        # 太高：上下裁
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    # 缩放到目标宽度
    new_h = int(width / target_ratio)
    return img.resize((width, new_h), Image.LANCZOS)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    file = request.files.get("image")
    if not file:
        return "No image uploaded", 400

    # 打开并转 RGB
    img = Image.open(file.stream).convert("RGB")

    # 预测分类
    preds = predict(img)
    label, conf = max(preds.items(), key=lambda x: x[1])
    conf_pct = round(conf * 100)

    # 生成简笔画
    sketch = to_sketch(
        img,
        low_thresh=80,
        high_thresh=150,
        thick_k=5,
        thin_k=2,
        noise_sigma=0.005,
        bin_thresh=0.5,
        r_ratio=0.5,
        sigma_ratio=1.0
    )

    # 裁剪＋缩放到 600×400 (3:2)
    sketch = crop_and_resize(sketch, target_ratio=3/2, width=600)

    # 转 DataURL
    result_src = pil_to_dataurl(sketch)

    # 获取文案
    info = disease_info.get(label, {
        "title":       label,
        "description": "暂无描述。",
        "tips":        "暂无建议。"
    })

    return render_template(
        "result.html",
        confidence=conf_pct,
        result_src=result_src,
        info=info
    )

if __name__ == "__main__":
    # 监听 8000 端口，任意网络地址
    app.run(host="0.0.0.0", port=8000, debug=True)
