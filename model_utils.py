# 模型下载、加载和预处理逻辑
# model_utils.py
# model_utils.py
import os
import torch
from torchvision import models
from huggingface_hub import hf_hub_download

# 把类别列表也放这里，保证 predict.py 能直接 import
class_names = [
    "Eczema",
    "Psoriasis",
    "Contact Dermatitis",
    "Tinea",
    "Herpes",
    "Lichen Planus",
    "Melasma",
    # …根据你的模型补全
]
num_classes = len(class_names)

def load_model():
    # 如果已经用 huggingface-cli login，这里就不需要传 token
    model_file = hf_hub_download(
        repo_id="abdlh/ResNet34_finetuned_for_skin_diseases_by-abdlh",
        filename="skin_model2.pth",
        use_auth_token=True
    )
    # 加载 state_dict
    state_dict = torch.load(model_file, map_location="cpu")
    # 构建网络结构
    model = models.resnet34(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # 将权重加载到模型里
    model.load_state_dict(state_dict)
    return model
