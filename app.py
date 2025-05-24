import io
import base64

from flask import Flask, render_template, request
from PIL import Image

from predict import predict
from cartoon_utils import to_animegan2  # 或者你现有的 to_cartoon_fast 等

app = Flask(__name__)

def pil_to_dataurl(img: Image.Image) -> str:
    """把 PIL Image 转成 `<img src="data:image/png;base64,..." />` 用的 DataURL"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

@app.route("/", methods=["GET"])
def index():
    # 第一页：只展示上传表单
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    # 接收上传的文件
    file = request.files.get("image")
    if not file:
        return "未上传图片", 400

    # 1. 打开成 PIL，跑分类
    img = Image.open(file.stream).convert("RGB")
    preds = predict(img)   # {'Eczema':0.7, ...}

    # 取 top1 标签
    label, conf = max(preds.items(), key=lambda x: x[1])
    conf_pct = round(conf * 100)

    # 2. 生成动漫化图
    anime_img = to_animegan2(img)  # PIL.Image

    # 3. 转 DataURL 嵌入到 HTML
    anime_dataurl = pil_to_dataurl(anime_img)

    return render_template(
        "result.html",
        label=label,
        confidence=conf_pct,
        anime_src=anime_dataurl
    )

if __name__ == "__main__":
    app.run(debug=True)
