# app.py

import gradio as gr
from predict import predict              # 你的皮肤病分类函数
from cartoon_utils import to_cartoon_sd  # 上面这个超速卡通化函数

with gr.Blocks(title="皮肤病识别＋极速卡通化 Demo") as demo:
    gr.Markdown("## 上传皮损图片，左侧显示分类结果，右侧显示快速卡通化结果")

    with gr.Row():
        inp     = gr.Image(type="pil", label="上传原图")
        cls_out = gr.Label(num_top_classes=3, label="分类结果")
        cart_out= gr.Image(label="卡通化结果")

    # 允许用户微调“卡通化强度”
    strength = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="卡通化强度 (strength)")

    def full_pipeline(img, s):
        res     = predict(img)
        cartoon = to_cartoon_sd(img, strength=s)
        return res, cartoon

    btn = gr.Button("Run")
    btn.click(
        fn=full_pipeline,
        inputs=[inp, strength],
        outputs=[cls_out, cart_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)
