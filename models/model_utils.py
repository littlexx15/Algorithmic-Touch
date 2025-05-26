# model_utils.py

import torch
from torchvision import models
from huggingface_hub import hf_hub_download

REPO_ID    = "abdlh/ResNet34_finetuned_for_skin_diseases_by-abdlh"
MODEL_FILE = "skin_model2.pth"

# —— 手动列出这 7 个类别 —— #
class_names = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Herpes HPV and other STDs Photos",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
]
num_classes = len(class_names)

def load_model():
    # 1. 下载权重到缓存
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE,
        use_auth_token=True
    )
    # 2. 载入 state_dict
    state_dict = torch.load(model_path, map_location="cpu")
    # 3. 定义网络结构
    model = models.resnet34(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # 4. 加载权重
    model.load_state_dict(state_dict)
    model.eval()
    return model
