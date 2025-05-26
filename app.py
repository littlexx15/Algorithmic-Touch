# app.py

import io
import base64

from flask import Flask, render_template, request
from PIL import Image

from predict import predict
from cartoon_utils import to_animegan2
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
    2. 二次元化
    3. 取文案
    4. 渲染模板
    """
    file = request.files.get("image")
    if not file:
        return "未上传图片", 400

    # 分类预测
    img = Image.open(file.stream).convert("RGB")
    preds = predict(img)  # {'Eczema':0.7, ...}
    label, conf = max(preds.items(), key=lambda x: x[1])
    conf_pct = round(conf * 100)

    # 二次元化
    anime_img = to_animegan2(img)
    anime_src = pil_to_dataurl(anime_img)

    # 文案
    info = disease_info.get(label, {
        "title":       label,
        "description": "抱歉，暂无此疾病的详细说明。",
        "tips":        "暂无护理建议。"
    })

    return render_template(
        "result.html",
        label=label,
        confidence=conf_pct,
        anime_src=anime_src,
        info=info
    )

if __name__ == "__main__":
    app.run(debug=True)
