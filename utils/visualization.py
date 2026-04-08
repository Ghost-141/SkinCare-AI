import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from core.logger import logger

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class YOLOClsWrapper(nn.Module):
    """Wrapper to handle YOLOv8 classification model output tuple."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # YOLOv8 classification returns (logits, features) or similar tuple
        return self.model(x)[0]


def get_model_architecture(model_path: str, num_classes: int = 10, device: str = "cpu"):
    """Reconstruct a regular nn.Module from a JIT or YOLO model path."""
    model_name = os.path.basename(model_path).lower()

    if "yolo" in model_name:
        from ultralytics import YOLO

        yolo_model = YOLO(model_path)
        # Extract the underlying PyTorch model
        model = yolo_model.model
        
        # Ensure gradients are enabled for Grad-CAM
        for param in model.parameters():
            param.requires_grad = True
            
        wrapped_model = YOLOClsWrapper(model)
        # Target the final convolution layer before pooling in the classification head
        if hasattr(model.model[9], "conv"):
            target_layers = [model.model[9].conv.conv]
        else:
            target_layers = [model.model[8]]

        wrapped_model.to(device)
        wrapped_model.eval()
        return wrapped_model, target_layers

    if "resnet" in model_name:
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layers = [model.layer4[-1]]
    elif "efficientnet" in model_name:
        model = models.efficientnet_b0()
        # Custom Classifier architecture for efficientNet_B0 Model
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(512, num_classes),
        )
        target_layers = [model.features[-1]]
    else:
        # Default fallback to ResNet50 if unsure
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layers = [model.layer4[-1]]

    try:
        jit_model = torch.jit.load(model_path, map_location=device)
        model.load_state_dict(jit_model.state_dict())
    except Exception as e:
        logger.error(f"Failed to load state dict from JIT model: {e}")

    model.to(device)
    model.eval()
    return model, target_layers


def generate_gradcam_plusplus(model, target_layers, input_tensor, target_class=None):
    """Generate advanced heatmap using GradCAM++."""
    try:
        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            # Specify the target class
            targets = None
            if target_class is not None:
                targets = [ClassifierOutputTarget(target_class)]

            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        return grayscale_cam[0, :]
    except Exception as e:
        logger.error(f"GradCAM++ generation error: {e}")
        return None


def create_and_save_heatmap(model_path: str, image_path: str, output_path: str, target_class: int = None):
    """Full pipeline using grad-cam library."""
    try:
        # Detect device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load image
        original_image = Image.open(image_path).convert("RGB")
        img_np = np.array(original_image).astype(np.float32) / 255.0

        # Preprocess for model
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = transform(original_image).unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        # Get model and target layers
        model, target_layers = get_model_architecture(
            model_path, num_classes=10, device=device
        )

        # Generate CAM
        grayscale_cam = generate_gradcam_plusplus(model, target_layers, input_tensor, target_class=target_class)

        if grayscale_cam is not None:
            cam_pil = Image.fromarray((grayscale_cam * 255).astype(np.uint8))
            cam_pil = cam_pil.resize(original_image.size, Image.BILINEAR)
            grayscale_cam_resized = np.array(cam_pil).astype(np.float32) / 255.0

            # Apply heatmap overlay
            visualization = show_cam_on_image(
                img_np, grayscale_cam_resized, use_rgb=True
            )

            # Save
            result_img = Image.fromarray(visualization)
            result_img.save(output_path)
            return output_path

        return None
    except Exception as e:
        logger.error(f"Heatmap generation error: {e}")
        return None
