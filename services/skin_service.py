import json
from pathlib import Path
import torch
import warnings

# Suppress TorchScript load deprecation warning
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="`torch.jit.load` is deprecated"
)

from torchvision import transforms
from PIL import Image
from services.interface.analysis_interface import ISkinAnalysis
from core.config import settings
from core.logger import logger
import os


class SkinService(ISkinAnalysis):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self.load_class_names("./data/class_mapping.json")
        self.current_model_path = settings.MODEL_PATH
        self.model_type = "jit"
        self.model = self.load_model(self.current_model_path)

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

            # Try loading as YOLO if "yolo" in the path
            if "yolo" in os.path.basename(model_path).lower():
                from ultralytics import YOLO

                model = YOLO(model_path)
                self.model_type = "yolo"
                logger.info(f"Successfully loaded YOLO model from {model_path}")
                return model

            # Load as TorchScript
            try:
                model = torch.jit.load(model_path, map_location=self.device)
                self.model_type = "jit"
                logger.info(f"Successfully loaded TorchScript model from {model_path}")
                model.to(self.device)
                model.eval()
                return model
            except Exception as e:
                # If jit fails, try ultralytics as fallback
                from ultralytics import YOLO

                model = YOLO(model_path)
                self.model_type = "yolo"
                logger.info(f"Fallen back to YOLO for {model_path}")
                return model

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e

    def list_models(self):
        model_dir = Path("models/weights")
        return [f.name for f in model_dir.glob("*.pt")]

    def load_model_by_name(self, model_name: str):
        model_path = os.path.join("models/weights", model_name)
        if os.path.exists(model_path):
            self.model = self.load_model(model_path)
            self.current_model_path = model_path
            return True
        return False

    def predict(self, image_path: str) -> tuple[str, float]:
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"Image path does not exist: {image_path}")

        try:
            if self.model_type == "yolo":
                results = self.model(image_path, verbose=False)
                if not results or len(results) == 0:
                    raise RuntimeError("YOLO model returned no results")

                result = results[0]
                probs = result.probs
                if probs is None:
                    raise RuntimeError("YOLO model results contain no probabilities")

                confidence = float(probs.top1conf.item())
                index = int(probs.top1)

                # Use class mapping from JSON if possible, otherwise clean YOLO names
                if index < len(self.classes):
                    prediction = self.classes[index]
                else:
                    # Fallback to YOLO names
                    raw_name = result.names[index]
                    import re

                    prediction = re.sub(r"^\d+\.\s*", "", raw_name)  # Strip prefixes

                return prediction, confidence

            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, index = torch.max(probabilities, 0)

            prediction = self.classes[index.item()]
            return prediction, confidence.item()
        except Exception as e:
            logger.error(f"Prediction Error for {image_path}: {str(e)}")
            raise RuntimeError(f"Model prediction failed: {str(e)}") from e
