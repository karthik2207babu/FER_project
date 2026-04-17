import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MappedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, mapping=None):
        super().__init__(root, transform=transform)
        self.mapping = mapping

    def __getitem__(self, index):
        path, _ = self.samples[index]
        folder_name = os.path.basename(os.path.dirname(path))
        target = self.mapping[folder_name] if self.mapping else _
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def get_dataloaders(data_dir, config, batch_size=32):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # We keep Grayscale(1) so the features match your best_emotion_model.pt
    tf = transforms.Compose([
        transforms.Grayscale(1), transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(), norm
    ])

    if config.name == "rafdb":
        train_path = os.path.join(data_dir, "DATASET/train")
        test_path = os.path.join(data_dir, "DATASET/test")
        
        train_ds = MappedImageFolder(train_path, tf, config.folder_to_idx)
        full_test_ds = MappedImageFolder(test_path, tf, config.folder_to_idx)
        
        # Creating a validation split from the test set for monitoring
        v_size = len(full_test_ds) // 2
        t_size = len(full_test_ds) - v_size
        val_ds, test_ds = torch.utils.data.random_split(full_test_ds, [v_size, t_size])
    else:
        train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), tf)
        val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), tf)
        test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), tf)

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2))