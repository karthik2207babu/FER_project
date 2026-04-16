from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FER2013_EMOTION_TO_IDX = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}


def get_transforms(image_size: int, is_train: bool):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomErasing(p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class FER2013Dataset(Dataset):
    def __init__(self, root_dir: Path, image_size: int = 224, is_train: bool = True) -> None:
        self.image_paths = []
        self.labels = []
        self.transform = get_transforms(image_size, is_train)

        for emotion_name, label_idx in FER2013_EMOTION_TO_IDX.items():
            emotion_dir = root_dir / emotion_name
            if not emotion_dir.exists():
                continue
            for img_path in emotion_dir.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.image_paths.append(img_path)
                    self.labels.append(label_idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]


class RAFDBDataset(Dataset):
    def __init__(self, root_dir: Path, image_size: int = 224, is_train: bool = True) -> None:
        root_dir = Path(root_dir)
        csv_name = "train_labels.csv" if is_train else "test_labels.csv"
        csv_path = root_dir / csv_name

        df = pd.read_csv(csv_path, header=None, names=["filename", "label"])

        if str(df.iloc[0]["label"]).strip().lower() == "label":
            df = df.iloc[1:].reset_index(drop=True)

        df["label"] = df["label"].astype(int) - 1

        print(f"🔍 Mapping ONLY aligned images inside {root_dir}...")

        # ✅ ONLY ALIGNED IMAGES (CRITICAL FIX)
        available_images = {}
        for img_path in root_dir.rglob("*_aligned.jpg"):
            available_images[img_path.name] = img_path

        print(f"🔍 Matching CSV with aligned images...")

        self.image_paths = []
        self.labels = []

        for _, row in df.iterrows():
            img_name = str(row["filename"]).strip()
            label = int(row["label"])

            aligned_name = img_name.replace(".jpg", "_aligned.jpg")

            if aligned_name in available_images:
                self.image_paths.append(available_images[aligned_name])
                self.labels.append(label)

        print(f"✅ Found {len(self.image_paths)} aligned images out of {len(df)}.")

        if len(self.image_paths) == 0:
            raise RuntimeError("CRITICAL ERROR: No aligned images found.")

        self.transform = get_transforms(image_size, is_train)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]


def create_dataloaders(
    dataset_type: str,
    root_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2
):

    root_dir = Path(root_dir)

    if dataset_type.upper() == "FER2013":
        train_dataset = FER2013Dataset(root_dir / "train", image_size, True)
        test_dataset = FER2013Dataset(root_dir / "test", image_size, False)

    elif dataset_type.upper() == "RAF-DB":
        train_dataset = RAFDBDataset(root_dir, image_size, True)
        test_dataset = RAFDBDataset(root_dir, image_size, False)

    else:
        raise ValueError("dataset_type must be either 'FER2013' or 'RAF-DB'")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader