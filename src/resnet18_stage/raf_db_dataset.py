from __future__ import annotations

from pathlib import Path

from torchvision import transforms
from torchvision.datasets import ImageFolder


EMOTION_LABELS = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happiness",
    "5": "Sadness",
    "6": "Anger",
    "7": "Neutral",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def project_root_from_file(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parents[2]


def resolve_default_data_root(project_root: Path) -> Path:
    mtcnn_root = project_root / "data" / "RAF-DB" / "MTCNN"
    dataset_root = project_root / "data" / "RAF-DB" / "DATASET"

    mtcnn_ready = (mtcnn_root / "train").exists() and (mtcnn_root / "test").exists()
    if mtcnn_ready:
        return mtcnn_root
    return dataset_root


def build_train_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class RAFDBImageFolder(ImageFolder):
    def __init__(self, root: str | Path, train: bool, image_size: int = 224) -> None:
        split = "train" if train else "test"
        dataset_root = Path(root).resolve() / split
        transform = build_train_transforms(image_size) if train else build_eval_transforms(image_size)
        super().__init__(root=str(dataset_root), transform=transform)

        self.split = split
        self.dataset_root = dataset_root
        self.emotion_names = [EMOTION_LABELS.get(class_name, class_name) for class_name in self.classes]

    @property
    def num_classes(self) -> int:
        return len(self.classes)
