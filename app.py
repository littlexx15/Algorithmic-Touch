import io
import base64

from flask import Flask, render_template, request
from PIL import Image

from predict import predict
from cartoon_utils import to_sketch  # 调用上面径向渐变版本
from disease_info import disease_info

app = Flask(__name__)

def pil_to_dataurl(img: Image.Image) -> str:
    """
    把 PIL Image 转成 <img src="data:image/png;base64,..." /> 所需的 DataURL
    """
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
    """
    结果页流程：
      1. 接收上传的图片并分类
      2. 生成简笔画
      3. 获取文案
      4. 渲染模板
    """
    file = request.files.get("image")
    if not file:
        return "No image uploaded", 400

    # 1. 打开并转 RGB
    img = Image.open(file.stream).convert("RGB")

    # 2. 皮肤病分类
    preds = predict(img)
    label, conf = max(preds.items(), key=lambda x: x[1])
    conf_pct = round(conf * 100)

    # 3. 生成简笔画
    result_img = to_sketch(
        img,
        low_thresh=50,
        high_thresh=150,
        thick_k=7,
        thin_k=3,
        noise_sigma=0.02
    )

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
