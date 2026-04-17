import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

class MappedImageFolder(datasets.ImageFolder):
    """Generalized folder loader that applies an optional label mapping."""
    def __init__(self, root, transform=None, mapping=None):
        super().__init__(root, transform=transform)
        self.mapping = mapping

    def __getitem__(self, index):
        path, _ = self.samples[index]
        if self.mapping:
            # Get the immediate parent folder name
            folder_name = os.path.basename(os.path.dirname(path))
            target = self.mapping.get(folder_name, _)
        else:
            target = _
        
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def get_dataloaders(data_dir, config, batch_size=32):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        norm
    ])

    # --- Smart Path Detection ---
    def find_path(names):
        for name in names:
            p = os.path.join(data_dir, name)
            if os.path.exists(p): return p
        return None

    # Look for Train
    train_path = find_path(['train', 'TRAIN', 'training', 'DATASET/train'])
    # Look for Val/Test
    val_path = find_path(['val', 'VAL', 'validation', 'test', 'TEST', 'DATASET/test'])
    # Look for a separate Test if it exists
    test_path = find_path(['test', 'TEST'])
    
    if not train_path or not val_path:
        raise FileNotFoundError(f"Could not find valid train/val folders in {data_dir}")

    # Use Mapped loader if a folder_to_idx exists in config, else standard behavior
    mapping = getattr(config, 'folder_to_idx', None)
    
    train_ds = MappedImageFolder(train_path, tf, mapping)
    val_ds = MappedImageFolder(val_path, tf, mapping)
    
    # If no separate test path, we use the val_ds as test_ds
    test_ds = MappedImageFolder(test_path, tf, mapping) if test_path else val_ds

    print(f"📦 Dataset: {config.name} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2))