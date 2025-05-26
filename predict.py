# predict.py
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from model_utils import load_model, class_names

model = load_model()
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(img: Image.Image):
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = np.argmax(probs)
    return { class_names[idx]: float(probs[idx]) }
