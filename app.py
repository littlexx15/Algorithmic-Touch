import io
from PIL import Image
from flask import Flask, render_template, request

from models.predict import predict
from utils.sketch_effects import to_sketch
from utils.disease_info import disease_info
from utils.image_helpers import pil_to_dataurl, crop_and_resize  # ✅ These two functions now come from an external module

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("pages/index.html")

@app.route("/result", methods=["POST"])
def result():
    file = request.files.get("image")
    if not file:
        return "No image uploaded", 400

    # Open and convert to RGB
    img = Image.open(file.stream).convert("RGB")

    # Predict classification
    preds = predict(img)
    label, conf = max(preds.items(), key=lambda x: x[1])
    conf_pct = round(conf * 100)

    # Generate sketch effect
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

    # Crop and resize to 600×400 (3:2 aspect ratio)
    sketch = crop_and_resize(sketch, target_ratio=3/2, width=600)

    # Convert to Data URL for embedding
    result_src = pil_to_dataurl(sketch)

    # Retrieve text content
    info = disease_info.get(label, {
        "title":       label,
        "description": "No description available.",
        "tips":        "No tips available."
    })

    return render_template(
        "pages/result.html",
        confidence=conf_pct,
        result_src=result_src,
        info=info
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
