import io
from PIL import Image
from flask import Flask, render_template, request

from models.predict import predict
from utils.sketch_effects import to_sketch
from utils.disease_info import disease_info
from utils.image_helpers import pil_to_dataurl, crop_and_resize  # ✅ 这两函数现在来自外部模块

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("pages/index.html")

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
        "pages/result.html",
        confidence=conf_pct,
        result_src=result_src,
        info=info
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
