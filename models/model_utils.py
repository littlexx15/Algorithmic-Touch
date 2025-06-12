import torch
from torchvision import models
from huggingface_hub import hf_hub_download

REPO_ID    = "abdlh/ResNet34_finetuned_for_skin_diseases_by-abdlh"
MODEL_FILE = "skin_model2.pth"

# —— Manually list the seven classes —— #
class_names = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis, Basal Cell Carcinoma, and Other Malignant Lesions",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Herpes, HPV, and Other STDs Photos",
    "Melanoma, Skin Cancer, Nevi, and Moles",
    "Nail Fungus and Other Nail Diseases",
]
num_classes = len(class_names)

def load_model():
    # 1. Download weights to cache
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE,
        use_auth_token=True
    )
    # 2. Load the state dictionary
    state_dict = torch.load(model_path, map_location="cpu")
    # 3. Define the network architecture
    model = models.resnet34(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # 4. Load the weights
    model.load_state_dict(state_dict)
    model.eval()
    return model
