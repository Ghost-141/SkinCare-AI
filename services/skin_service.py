import json
from pathlib import Path
import torch
import warnings

# Suppress TorchScript load deprecation warning
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="`torch.jit.load` is deprecated"
)

from torchvision import models, transforms
from PIL import Image
from services.interface.analysis_interface import ISkinAnalysis
from core.config import settings
from core.logger import logger
import os


class SkinService(ISkinAnalysis):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self.load_class_names("./data/class_mapping.json")
        self.model = self.load_model(settings.MODEL_PATH)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def load_class_names(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def load_model(self, model_path: str):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Critical: Model file not found at {model_path}"
                )

            # Load as TorchScript - This handles architecture automatically
            model = torch.jit.load(model_path, map_location=self.device)
            logger.info(f"Successfully loaded TorchScript model from {model_path}")

            model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            logger.error(f"Failed to load JIT model: {str(e)}")
            raise e

    def list_models(self):
        model_dir = Path("models/weights")
        return [f.name for f in model_dir.glob("*.pt")]

    def load_model_by_name(self, model_name: str):
        model_path = os.path.join("models/weights", model_name)
        if os.path.exists(model_path):
            self.model = self.load_model(model_path)
            return True
        return False

    def predict(self, image_path: str) -> tuple[str, float]:
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, index = torch.max(probabilities, 0)

            prediction = self.classes[index.item()]
            return prediction, confidence.item()
        except Exception as e:
            logger.error(f"Prediction Error: {str(e)}")
            return "Unknown", 0.0
