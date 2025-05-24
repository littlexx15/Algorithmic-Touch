# app.py

import gradio as gr
from predict import predict              # 你的皮肤病分类函数
from cartoon_utils import to_cartoon_fast

with gr.Blocks(title="皮肤病识别＋极速画风转换 Demo") as demo:
    gr.Markdown("## 上传皮损图片，左侧显示分类结果，右侧显示画风转换结果")

    with gr.Row():
        inp     = gr.Image(type="pil", label="上传原图")
        cls_out = gr.Label(num_top_classes=3, label="分类结果")
        cart_out= gr.Image(label="画风转换结果")

    # 允许用户调节两个参数，看看不同细节/边缘保留效果
    sigma_s = gr.Slider(0, 200, value=60, step=10, label="细节保留 (sigma_s)")
    sigma_r = gr.Slider(0.0, 1.0, value=0.07, step=0.01, label="边缘保留 (sigma_r)")

    def full_pipeline(img, s_s, s_r):
        # 1. 分类
        res     = predict(img)
        # 2. OpenCV stylization 卡通化
        cartoon = to_cartoon_fast(img, sigma_s=s_s, sigma_r=s_r)
        return res, cartoon

    btn = gr.Button("Run")
    btn.click(
        fn=full_pipeline,
        inputs=[inp, sigma_s, sigma_r],
        outputs=[cls_out, cart_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)
