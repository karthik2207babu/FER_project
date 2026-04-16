from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet18

# Force matplotlib to run headlessly so it doesn't crash Colab
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import your custom modules
from dataset_stage.fer_dataset import create_dataloaders
from lfa_stage.lfa_module import LocalFeatureAugmentation
from msgc_stage.msgc_module import MultiScaleGlobalConvolution
from safm_stage.safm_module import SpatialAttentionFeatureModule
from tokenization_stage.tokenization_module import RegionTokenization
from frit_stage.frit_module import FRITTransformer
from classification_stage.classification_module import EmotionClassifier


class FERFullPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load the ImageNet weights
        self.backbone = resnet18(weights='DEFAULT') 
        
        # THE FIX: Freeze the backbone so the Transformer doesn't destroy it!
        for name, param in self.backbone.named_parameters():
            if "layer1" in name or "layer2" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
        self.lfa = LocalFeatureAugmentation(channels=128)
        self.msgc = MultiScaleGlobalConvolution(channels=128)
        self.safm = SpatialAttentionFeatureModule()
        self.tokenization = RegionTokenization()
        self.frit = FRITTransformer(input_dim=128, embed_dim=64)
        
        self.dropout = nn.Dropout(p=0.5) 
        
        self.classifier = EmotionClassifier(embed_dim=64, num_classes=7)
    def forward(self, x):
    # ResNet backbone upto layer2
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)

        # pipeline
        x = self.lfa(x)
        x = self.msgc(x)
        x = self.safm(x)

        tokens = self.tokenization(x)
        t_prime = self.frit(tokens)
        t_prime = self.dropout(t_prime)

        logits = self.classifier(t_prime)

        return logits

# The Live Plotting Function
def save_live_plot(history: dict, save_path: Path):
    """Draws a detailed graph with points for every epoch and saves it."""
    plt.figure(figsize=(14, 6))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], marker='o', linestyle='-', color='blue', label='Train Accuracy')
    plt.plot(history['val_acc'], marker='o', linestyle='-', color='green', label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], marker='o', linestyle='-', color='red', label='Train Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # Free up memory


def train_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", type=str, default="RAF-DB")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50) 
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--save-dir", type=Path, default=Path("/content/drive/MyDrive/FER_Checkpoints"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting High-Accuracy Training (ImageNet Backbone) on: {device}")
    
    args.save_dir.mkdir(parents=True, exist_ok=True)
    graph_path = args.save_dir / "live_learning_curve.png"

    # In RAF-DB, the 'test' folder is strictly used as our Validation Set during training.
    train_loader, val_loader = create_dataloaders(
        dataset_type=args.dataset_type,
        root_dir=args.data_dir,
        batch_size=args.batch_size
    )

    model = FERFullPipeline().to(device)
    counts = Counter(train_loader.dataset.labels)
    total = sum(counts.values())
    weights = torch.tensor([total / counts[i] for i in range(len(counts))]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(
    [
        {"params": model.backbone.layer2.parameters(), "lr": args.lr},
        {"params": model.lfa.parameters(), "lr": args.lr},
        {"params": model.msgc.parameters(), "lr": args.lr},
        {"params": model.safm.parameters(), "lr": args.lr},
        {"params": model.frit.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr * 5},
    ],
    weight_decay=1e-4
)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    
    # Early Stopping Trackers
    early_stop_patience = 7
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 20)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation Loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        # Update history and redraw the graph instantly!
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        save_live_plot(history, graph_path)
        print(f"📊 Graph updated at: {graph_path.name}")

        # Save Checkpoint & Early Stopping Logic
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_without_improvement = 0 # Reset the clock
            save_path = args.save_dir / f"pure_{args.dataset_type.lower()}_from_scratch.pt"
            torch.save(model.state_dict(), save_path)
            print(f"⭐ New best model saved! ({best_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s).")
            
            if epochs_without_improvement >= early_stop_patience:
                print(f"\n🛑 EARLY STOPPING TRIGGERED! The model stopped learning general rules. Halting to save your best weights.")
                break # Kills the loop

if __name__ == "__main__":
    train_model()