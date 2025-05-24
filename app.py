# app.py

import io
import base64

from flask import Flask, render_template, request
from PIL import Image

from predict import predict
from cartoon_utils import to_animegan2

app = Flask(__name__)

def pil_to_dataurl(img: Image.Image) -> str:
    """把 PIL Image 转成 <img src="data:image/png;base64,..." /> 所需的 DataURL"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

@app.route("/", methods=["GET"])
def index():
    # 第一页：上传表单
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    # 接收上传的文件
    file = request.files.get("image")
    if not file:
        return "未上传图片", 400

    # 1. PIL 打开 & 诊断分类
    img = Image.open(file.stream).convert("RGB")
    preds = predict(img)   # 返回 dict: {label: prob, ...}

    # 取 top1 标签 和 置信度%
    label, conf = max(preds.items(), key=lambda x: x[1])
    conf_pct = round(conf * 100)

    # 2. AnimeGAN2 动漫化
    anime_img = to_animegan2(img)

    # 3. 转 DataURL 嵌入到 HTML
    anime_dataurl = pil_to_dataurl(anime_img)

    return render_template(
        "result.html",
        label=label,
        confidence=conf_pct,
        anime_src=anime_dataurl
    )

if __name__ == "__main__":
    # debug=True 只在开发时用，生产/部署请去掉
    app.run(debug=True)
