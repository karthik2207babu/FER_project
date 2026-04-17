import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Modular imports
from model import FERFullPipeline
from dataset_stage.fer_dataset import get_dataloaders
from configs import get_dataset_config 

def save_plot(history, save_dir, dataset_name):
    """Saves learning curves to monitor for overfitting."""
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc', color='blue', marker='o')
    plt.plot(history['val_acc'], label='Val Acc', color='green', marker='o')
    plt.title(f'Accuracy ({dataset_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', color='red', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', color='orange', marker='o')
    plt.title(f'Loss ({dataset_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_learning_curve.png"))
    plt.close()

def train_model(model, train_loader, val_loader, test_loader, device, epochs, lr, save_dir, cfg):
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dataset-specific weights from our Config class
    weights = cfg.weights.to(device) if cfg.weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Early Stopping Setup
    patience_limit = 8
    no_improve_counter = 0
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        start_time = time.time()
        
        # --- 1. Training Phase ---
        model.train()
        running_loss, running_corrects = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = (running_corrects.double() / len(train_loader.dataset)).item()

        # --- 2. Validation Phase ---
        model.eval()
        v_loss, v_corrects = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                v_loss += loss.item() * images.size(0)
                v_corrects += torch.sum(preds == labels.data)
    
        epoch_val_loss = v_loss / len(val_loader.dataset)
        epoch_val_acc = (v_corrects.double() / len(val_loader.dataset)).item()

        # Update Scheduler & History
        scheduler.step(epoch_val_acc)
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train: {epoch_train_acc*100:.2f}% | Val: {epoch_val_acc*100:.2f}% | "
              f"LR: {current_lr:.7f} | Time: {time.time()-start_time:.1f}s")

        save_plot(history, save_dir, cfg.name)

        # Checkpoint & Early Stopping
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_{cfg.name}_model.pt"))
            print(f"⭐ New Best {cfg.name.upper()} Model Saved!")
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience_limit:
                print(f"🛑 Early stopping triggered after {patience_limit} epochs of no improvement.")
                break

    # --- 3. Final Test Evaluation ---
    print(f"\n🔍 Final Evaluation on {cfg.name} Test Set...")
    model.load_state_dict(best_model_wts)
    model.eval()
    t_corrects = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            t_corrects += torch.sum(preds == labels.data)
    
    final_test_acc = (t_corrects.double() / len(test_loader.dataset)).item()
    print(f"✅ Final {cfg.name} Test Accuracy: {final_test_acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="balanced", help="balanced or rafdb")
    parser.add_argument("--resume-path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--save-dir", type=str, default="/content/drive/MyDrive/FER_Results")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Dataset-Specific Config
    cfg = get_dataset_config(args.dataset)

    # Setup Dataloaders
    data_path = Path(args.data_dir)
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
        data_path / "train", data_path / "val", data_path / "test", args.batch_size
    )

    # Initialize Architecture
    model = FERFullPipeline(num_classes=cfg.num_classes).to(DEVICE)

    # Logic to load weights for Fine-Tuning
    if args.resume_path and os.path.exists(args.resume_path):
        print(f"🚀 Fine-tuning mode: Loading weights from {args.resume_path}")
        model.load_state_dict(torch.load(args.resume_path, map_location=DEVICE, weights_only=True))

    train_model(model, train_loader, val_loader, test_loader, DEVICE, args.epochs, args.lr, args.save_dir, cfg)