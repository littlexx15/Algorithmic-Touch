import torch
from PIL import Image
from torchvision import transforms
import numpy as np

from models.model_utils import load_model, class_names

# —— Load the model only once when the module is imported —— #
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
    Takes a PIL.Image as input and returns a dictionary mapping
    each class name to its probability:
      {class_name: prob, ...}
    """
    x = preprocess(img).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    # Return the full mapping rather than just the top class
    return {name: float(prob) for name, prob in zip(class_names, probs)}
