import os, time, copy, torch, argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

from model import FERFullPipeline
from dataset_stage.fer_dataset import get_dataloaders
from configs import get_dataset_config 

def save_plot(history, save_dir, name):
    if not history['train_acc']: return # Don't plot if no training happened
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc'); plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'Acc - {name}'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss'); plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Loss - {name}'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{name}_learning_curves.png"))
    plt.close()

def train_model(model, loaders, device, epochs, lr, save_dir, cfg, resume_path):
    os.makedirs(save_dir, exist_ok=True)
    train_loader, val_loader, test_loader = loaders
    
    criterion = nn.CrossEntropyLoss(weight=cfg.weights.to(device) if cfg.weights is not None else None, label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_acc, history = 0.0, {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # Only run training if epochs > 0
    if epochs > 0:
        for epoch in range(epochs):
            model.train()
            r_loss, r_corr = 0.0, 0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad(); out = model(imgs); loss = criterion(out, lbls)
                _, preds = torch.max(out, 1); loss.backward(); optimizer.step()
                r_loss += loss.item() * imgs.size(0); r_corr += torch.sum(preds == lbls.data)

            model.eval()
            v_loss, v_corr = 0.0, 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    out = model(imgs); loss = criterion(out, lbls); _, preds = torch.max(out, 1)
                    v_loss += loss.item() * imgs.size(0); v_corr += torch.sum(preds == lbls.data)
        
            t_acc, v_acc = (r_corr.double()/len(train_loader.dataset)).item(), (v_corr.double()/len(val_loader.dataset)).item()
            scheduler.step(v_acc)
            history['train_loss'].append(r_loss/len(train_loader.dataset)); history['val_loss'].append(v_loss/len(val_loader.dataset))
            history['train_acc'].append(t_acc); history['val_acc'].append(v_acc)

            print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {t_acc*100:.2f}% | Val Acc: {v_acc*100:.2f}% | LR: {optimizer.param_groups[0]['lr']:.7f}")
            save_plot(history, save_dir, cfg.name)

            if v_acc > best_acc:
                best_acc = v_acc
                torch.save(model.state_dict(), os.path.join(save_dir, f"best_{cfg.name}_model.pt"))
                print(f"⭐ New Best {cfg.name} weights saved!")

    # --- FINAL EVALUATION ---
    print(f"\n🔍 Evaluating Model on {cfg.name} Test Set...")
    
    # Logic: If we just trained, load the new best. If we skipped training, load the resume_path.
    load_path = os.path.join(save_dir, f"best_{cfg.name}_model.pt") if epochs > 0 else resume_path
    
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
        model.eval(); t_corr = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device); out = model(imgs); _, preds = torch.max(out, 1)
                t_corr += torch.sum(preds == lbls.data)
        print(f"✅ Final {cfg.name} Test Accuracy: {(t_corr.double()/len(test_loader.dataset))*100:.2f}%")
    else:
        print(f"❌ Error: Could not find weights at {load_path} for testing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="rafdb")
    parser.add_argument("--resume-path", type=str, default="/content/drive/MyDrive/best_emotion_model.pt")
    parser.add_argument("--save-dir", type=str, default="/content/drive/MyDrive/Results")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_dataset_config(args.dataset)
    loaders = get_dataloaders(args.data_dir, cfg)
    
    model = FERFullPipeline(num_classes=cfg.num_classes).to(DEVICE)
    if os.path.exists(args.resume_path):
        print(f"🚀 Loading weights from: {args.resume_path}")
        model.load_state_dict(torch.load(args.resume_path, map_location=DEVICE, weights_only=True))

    train_model(model, loaders, DEVICE, args.epochs, args.lr, args.save_dir, cfg, args.resume_path)