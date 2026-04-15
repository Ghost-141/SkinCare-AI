import json
from pathlib import Path
import torch
import warnings
import os
from PIL import Image
from torchvision import transforms
from services.interface.analysis_interface import ISkinAnalysis
from core.config import settings
from core.logger import logger

# Suppress TorchScript load deprecation warning
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="`torch.jit.load` is deprecated"
)

class SkinService(ISkinAnalysis):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self.load_class_names("./data/class_mapping.json")
        
        # Dictionary to store pre-loaded models
        self.models = {}
        self.current_model_name = "EfficientNet_B0.pt"
        
        # Pre-load all available models for instant switching
        self.preload_all_models()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def preload_all_models(self):
        """Load all .pt files in models/weights into memory on startup."""
        model_dir = Path("models/weights")
        for model_path in model_dir.glob("*.pt"):
            name = model_path.name
            try:
                logger.info(f"Pre-loading model: {name}...")
                self.models[name] = self.load_model(str(model_path))
            except Exception as e:
                logger.error(f"Failed to pre-load {name}: {e}")

    def load_model(self, model_path: str):
        """Internal helper to load a model based on its type."""
        # Case 1: YOLO
        if "yolo" in os.path.basename(model_path).lower():
            from ultralytics import YOLO
            return {"type": "yolo", "instance": YOLO(model_path)}

        # Case 2: TorchScript (EfficientNet, ResNet)
        try:
            model = torch.jit.load(model_path, map_location=self.device)
            model.to(self.device)
            model.eval()
            return {"type": "jit", "instance": model}
        except Exception as e:
            # Fallback for complex weights
            from ultralytics import YOLO
            return {"type": "yolo", "instance": YOLO(model_path)}

    def load_class_names(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def list_models(self):
        return list(self.models.keys())

    def load_model_by_name(self, model_name: str):
        """Instant switch between pre-loaded models."""
        if model_name in self.models:
            self.current_model_name = model_name
            logger.info(f"Switched active model to: {model_name}")
            return True
        return False

    def predict(self, image_path: str) -> tuple[str, float, int]:
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"Image path does not exist: {image_path}")

        model_entry = self.models.get(self.current_model_name)
        if not model_entry:
            raise RuntimeError(f"Active model '{self.current_model_name}' not loaded.")

        model_type = model_entry["type"]
        model = model_entry["instance"]

        try:
            if model_type == "yolo":
                results = model(image_path, verbose=False)
                result = results[0]
                probs = result.probs
                confidence = float(probs.top1conf.item())
                index = int(probs.top1)
                
                if index < len(self.classes):
                    prediction = self.classes[index]
                else:
                    prediction = result.names[index]
                return prediction, confidence, index

            # TorchScript Inference
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, index = torch.max(probabilities, 0)

            index_val = index.item()
            prediction = self.classes[index_val]
            return prediction, confidence.item(), index_val

        except Exception as e:
            logger.error(f"Prediction Error for {image_path}: {str(e)}")
            raise RuntimeError(f"Model prediction failed: {str(e)}") from e
