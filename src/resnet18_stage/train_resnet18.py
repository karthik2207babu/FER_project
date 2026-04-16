from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from resnet18_stage.raf_db_dataset import RAFDBImageFolder, project_root_from_file, resolve_default_data_root


def build_parser() -> argparse.ArgumentParser:
    project_root = project_root_from_file(__file__)
    default_data_root = resolve_default_data_root(project_root)
    default_output_dir = project_root / "outputs" / "resnet18"

    parser = argparse.ArgumentParser(
        description="Train a ResNet-18 classifier on RAF-DB train/test folders."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help="Dataset root containing train/ and test/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Where checkpoints and metrics will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use ImageNet pretrained ResNet-18 weights or start from scratch.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use GPU automatically if available, or force CPU/CUDA.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 on Windows if multiprocessing causes issues.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the final classifier layer.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional debug limit for train batches per epoch.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional debug limit for validation batches per epoch.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(num_classes: int, weights_mode: str, freeze_backbone: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if weights_mode == "imagenet" else None
    model = resnet18(weights=weights)

    if freeze_backbone:
     for name, param in model.named_parameters():
        if "layer1" in name or "layer2" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def create_dataloaders(
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, RAFDBImageFolder, RAFDBImageFolder]:
    train_dataset = RAFDBImageFolder(root=data_root, train=True, image_size=image_size)
    val_dataset = RAFDBImageFolder(root=data_root, train=False, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

            if max_batches is not None and batch_index >= max_batches:
                break

    average_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return average_loss, accuracy


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_accuracy: float,
    class_names: list[str],
    emotion_names: list[str],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "class_names": class_names,
            "emotion_names": emotion_names,
            "args": vars(args),
        },
        path,
    )


def validate_dataset_layout(data_root: Path) -> None:
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected train/ and test/ inside {data_root}, but one or both are missing."
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()

    validate_dataset_layout(args.data_root)
    seed_everything(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Training data root: {args.data_root}")
    print(f"Outputs will be saved to: {args.output_dir}")

    pin_memory = device.type == "cuda"
    train_loader, val_loader, train_dataset, _ = create_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    counts = Counter(train_dataset.targets)
    total = sum(counts.values())
    weights = [total / counts[i] for i in range(len(counts))]
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    model = create_model(
        num_classes=train_dataset.num_classes,
        weights_mode=args.weights,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
    [
        {"params": model.layer2.parameters(), "lr": args.lr},
        {"params": model.fc.parameters(), "lr": args.lr * 10},
    ],
    weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_loss, val_accuracy = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )
        scheduler.step()

        epoch_seconds = time.time() - start_time
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "seconds": epoch_seconds,
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} | "
            f"time={epoch_seconds:.1f}s"
        )

        save_checkpoint(
            path=args.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_accuracy=max(best_val_accuracy, val_accuracy),
            class_names=train_dataset.classes,
            emotion_names=train_dataset.emotion_names,
            args=args,
        )
        save_json(args.output_dir / "history.json", {"history": history})

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                path=args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                class_names=train_dataset.classes,
                emotion_names=train_dataset.emotion_names,
                args=args,
            )

    summary = {
        "best_val_accuracy": best_val_accuracy,
        "epochs": args.epochs,
        "data_root": str(args.data_root),
        "classes": train_dataset.classes,
        "emotion_names": train_dataset.emotion_names,
    }
    save_json(args.output_dir / "summary.json", summary)

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best checkpoint: {args.output_dir / 'best.pt'}")
    print(f"Training history: {args.output_dir / 'history.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
