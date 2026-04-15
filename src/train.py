from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torchvision.models import resnet18

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
    """Wraps all your modules into one clean model for training."""
    def __init__(self):
        super().__init__()
        # 1. Backbone (Initialized with random weights for training)
        self.backbone = resnet18(weights=None)
        
        # 2. Custom Modules
        self.lfa = LocalFeatureAugmentation(channels=128)
        self.msgc = MultiScaleGlobalConvolution(channels=128)
        self.safm = SpatialAttentionFeatureModule()
        self.tokenization = RegionTokenization()
        self.frit = FRITTransformer(input_dim=128, embed_dim=64)
        self.classifier = EmotionClassifier(embed_dim=64, num_classes=7)

    def forward(self, x):
        # ResNet up to layer 2 (Outputs 128x28x28)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        
        # Custom Pipeline
        x = self.lfa(x)
        x = self.msgc(x)
        x = self.safm(x)
        tokens = self.tokenization(x)
        frit_out = self.frit(tokens)
        logits = self.classifier(frit_out)
        
        return logits


def train_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", type=str, required=True, choices=["FER2013", "RAF-DB"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001) # Lowered slightly for more stable early training
    parser.add_argument("--save-dir", type=Path, default=Path("/content/drive/MyDrive/FER_Checkpoints"))
    args = parser.parse_args()

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting training on: {device}")
    
    # Create save directory in your Google Drive
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Load Data
    print(f"📦 Loading {args.dataset_type} from {args.data_dir}...")
    train_loader, test_loader = create_dataloaders(
        dataset_type=args.dataset_type,
        root_dir=args.data_dir,
        batch_size=args.batch_size
    )

    # Initialize Model, Loss, and Optimizer
    model = FERFullPipeline().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0

    # The Training Loop
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

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        
        # Evaluation Loop
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        print(f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # Save Checkpoint to Google Drive
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = args.save_dir / f"best_{args.dataset_type.lower()}_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"⭐ New best model saved to {save_path}!")

if __name__ == "__main__":
    train_model()