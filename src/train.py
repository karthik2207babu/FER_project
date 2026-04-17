import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Assuming your assembled model is in src/model.py
from model import FERFullPipeline
from dataset_stage.fer_dataset import get_dataloaders

def save_plot(history, save_dir):
    """Saves high-accuracy learning curves to Google Drive every epoch."""
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc', color='blue', marker='o')
    plt.plot(history['val_acc'], label='Val Acc', color='green', marker='o')
    plt.title('Epoch vs Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', color='red', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', color='orange', marker='o')
    plt.title('Epoch vs Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_curve.png"))
    plt.close()

def train_model(model, train_loader, val_loader, test_loader, device, epochs, lr, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        running_corrects = 0

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

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels.data)
    
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = (val_corrects.double() / len(val_loader.dataset)).item()

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {epoch_train_acc:.4f} | "
              f"Val Acc: {epoch_val_acc:.4f} | Time: {time.time() - start_time:.1f}s")

        save_plot(history, save_dir)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(save_dir, "best_emotion_model.pt"))
            print(f"⭐ New Best Model Saved!")

    print("\n🔍 Final Evaluation on the Test Set...")
    model.load_state_dict(best_model_wts)
    model.eval()
    test_corrects = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
    
    print(f"✅ Final Test Accuracy: {(test_corrects.double() / len(test_loader.dataset)):.4f}")
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Fixed: Only one data-dir required now
    parser.add_argument("--data-dir", type=str, required=True, help="Path to folder containing train/val/test")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--save-dir", type=str, default="/content/drive/MyDrive/FER_Checkpoints")
    args = parser.parse_args()

    # Logic to automatically find the subfolders
    base_path = Path(args.data_dir)
    train_p = base_path / "train"
    val_p = base_path / "val"
    test_p = base_path / "test"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
        train_p, val_p, test_p, args.batch_size
    )

    model = FERFullPipeline(num_classes=7).to(DEVICE)
    train_model(model, train_loader, val_loader, test_loader, DEVICE, args.epochs, args.lr, args.save_dir)