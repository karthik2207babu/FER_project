from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ResNet-18 ImageNet normalization requirements
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# FER2013 standard folder mapping
FER2013_EMOTION_TO_IDX = {
    "surprise": 0, "fear": 1, "disgust": 2, "happy": 3, 
    "sad": 4, "angry": 5, "neutral": 6
}

def get_transforms(image_size: int, is_train: bool):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class FER2013Dataset(Dataset):
    """Handles the FER2013 train/test folder structure."""
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]


class RAFDBDataset(Dataset):
    """Handles the RAF-DB CSV mapping structure."""
    def __init__(self, root_dir: Path, image_size: int = 224, is_train: bool = True) -> None:
        self.img_dir = root_dir / "DATASET"
        csv_name = "train_labels.csv" if is_train else "test_labels.csv"
        csv_path = root_dir / csv_name
        
        # Read CSV. (RAF-DB labels are 1-7. We subtract 1 to make them 0-6 to match FER2013)
        self.df = pd.read_csv(csv_path, header=None, names=["filename", "label"])
        self.df["label"] = self.df["label"] - 1 
        
        self.transform = get_transforms(image_size, is_train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_name = str(self.df.iloc[idx]["filename"])
        label = int(self.df.iloc[idx]["label"])
        
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert("RGB")
        
        return self.transform(image), label


def create_dataloaders(
    dataset_type: str,
    root_dir: str | Path, 
    batch_size: int = 32, 
    image_size: int = 224,
    num_workers: int = 2
) -> tuple[DataLoader, DataLoader]:
    
    root_dir = Path(root_dir)
    
    if dataset_type.upper() == "FER2013":
        train_dataset = FER2013Dataset(root_dir / "train", image_size, is_train=True)
        test_dataset = FER2013Dataset(root_dir / "test", image_size, is_train=False)
    elif dataset_type.upper() == "RAF-DB":
        train_dataset = RAFDBDataset(root_dir, image_size, is_train=True)
        test_dataset = RAFDBDataset(root_dir, image_size, is_train=False)
    else:
        raise ValueError("dataset_type must be either 'FER2013' or 'RAF-DB'")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader