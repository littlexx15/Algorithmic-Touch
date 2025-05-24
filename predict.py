# # 封装 predict(img) 函数
# predict.py
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from model_utils import load_model, class_names  # 同时 import class_names

# 加载模型
model = load_model()
model.eval()

# 与训练时同样的预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(img: Image.Image):
    """
    输入：PIL.Image
    输出：{类别名: 置信度}
    """
    # 预处理并增加 batch 维度
    x = preprocess(img).unsqueeze(0)        # shape: [1,3,224,224]
    with torch.no_grad():
        logits = model(x)                   # 前向
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = np.argmax(probs)
    return { class_names[idx]: float(probs[idx]) }
