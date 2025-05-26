# predict.py

import torch
from PIL import Image
from torchvision import transforms
import numpy as np

from model_utils import load_model, class_names

# —— 只在模块导入时加载一次模型 —— #
model = load_model()
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def predict(img: Image.Image) -> dict[str, float]:
    """
    输入 PIL.Image，返回每个类别的概率字典：
      {class_name: prob, ...}
    """
    x = preprocess(img).unsqueeze(0)  # 添加 batch 维度
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    # 返回完整映射，而不是只返回最高项
    return {name: float(prob) for name, prob in zip(class_names, probs)}
