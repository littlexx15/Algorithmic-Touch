# app.py

import io
import base64

from flask import Flask, render_template, request
from PIL import Image

from predict import predict
from cartoon_utils import to_animegan2, to_sketch
from disease_info import disease_info

app = Flask(__name__)

def pil_to_dataurl(img: Image.Image) -> str:
    """把 PIL Image 转成 <img src="data:image/png;base64,..." /> 所需的 DataURL"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

@app.route("/", methods=["GET"])
def index():
    """上传页：只渲染表单"""
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    """结果页：
    1. 接收上传的图片、跑分类
    2. 二次元化 or 简笔画
    3. 取文案
    4. 渲染模板
    """
    file = request.files.get("image")
    if not file:
        return "No image uploaded", 400

    # 1. 打开并转 RGB
    img = Image.open(file.stream).convert("RGB")

    # 2. 分类预测
    preds = predict(img)                # e.g. {'Eczema':0.7, ...}
    label, conf = max(preds.items(), key=lambda x: x[1])
    conf_pct = round(conf * 100)

    # 3. 选择效果：AnimeGANv2 或 简笔画
    # —— 如果要动漫化：
    # result_img = to_animegan2(img)
    # —— 如果要简笔画边缘：
    result_img = to_sketch(img, low_threshold=50, high_threshold=150)

    # 4. 转成 DataURL
    result_src = pil_to_dataurl(result_img)

    # 5. 文案信息
    info = disease_info.get(label, {
        "title":       label,
        "description": "Sorry, no detailed description available.",
        "tips":        "No care tips available."
    })

    # 6. 渲染页面
    return render_template(
        "result.html",
        title="Algorithmic Touch",
        confidence=conf_pct,
        result_src=result_src,
        info=info
    )

if __name__ == "__main__":
    app.run(debug=True)
