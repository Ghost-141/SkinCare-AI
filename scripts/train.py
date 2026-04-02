import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import os
import argparse
from tqdm import tqdm
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser(description="Skin Disease Training Script")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset (IMG_CLASSES folder)")
    parser.add_argument("--model_type", type=str, default="resnet", choices=["resnet", "efficientnet"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="scripts/output", help="Directory to save models and plots")
    return parser.parse_args()

def get_model(choice, num_classes=10, device="cpu"):
    if choice == "resnet":
        model = models.resnet50(weights='IMAGENET1K_V2')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif choice == "efficientnet":
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    for i, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        current_acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{running_loss/(i+1):.4f}', 'acc': f'{current_acc:.2f}%'})
    return running_loss / len(loader), 100. * correct / total

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Validating", leave=False)
    with torch.no_grad():
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'val_loss': f'{running_loss/(i+1):.4f}'})
    return running_loss / len(loader), 100. * correct / total

def plot_history(history, output_dir):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue', linestyle='--')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='blue', linestyle='--')
    plt.plot(history['val_acc'], label='Val Acc', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()

def plot_confusion_matrix(model, loader, class_names, device, output_dir):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    clean_names = [name.split('. ')[1].split(' ')[0] if '. ' in name else name for name in class_names]
    
    plt.figure(figsize=(14, 10))
    sns.set_context("paper", font_scale=1.2)
    ax = sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=clean_names, yticklabels=clean_names,
        annot_kws={"size": 10}, cbar_kws={'label': 'Number of Images'}
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Skin Disease Classification Performance\n(Confusion Matrix)', fontsize=16, pad=20)
    plt.xlabel('Predicted Diagnosis', fontsize=14, labelpad=10)
    plt.ylabel('Actual Diagnosis', fontsize=14, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model-specific output directory
    model_output_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(model_output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Sampler and class weights
    train_indices = train_dataset.indices
    train_labels = np.array([full_dataset.targets[i] for i in train_indices])
    unique_classes, class_sample_count = np.unique(train_labels, return_counts=True)
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(weight[train_labels]).double()
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
    
    class_weights = 1.0 / torch.tensor(class_sample_count, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * len(unique_classes)
    class_weights = class_weights.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Save class mapping
    class_indices = full_dataset.class_to_idx
    classes = [None] * len(class_indices)
    for name, index in class_indices.items():
        clean_name = name.split('. ')[1] if '. ' in name else name
        classes[index] = clean_name
    with open(os.path.join(model_output_dir, 'class_mapping.json'), 'w') as f:
        json.dump(classes, f, indent=4)
    
    model = get_model(args.model_type, num_classes=len(classes), device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{args.epochs}: [Train] Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | [Val] Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
    
    # Save model and Jit
    torch.save(model.state_dict(), os.path.join(model_output_dir, f"skin_{args.model_type}.pth"))
    model.eval()
    example_input = torch.rand(1, 3, 224, 224).to(device)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(os.path.join(model_output_dir, f"skin_{args.model_type}_jit.pt"))
    
    # Plotting
    print("Generating plots...")
    plot_history(history, model_output_dir)
    plot_confusion_matrix(model, val_loader, full_dataset.classes, device, model_output_dir)
    print(f"Training complete. Outputs saved in {model_output_dir}")

if __name__ == "__main__":
    main()
