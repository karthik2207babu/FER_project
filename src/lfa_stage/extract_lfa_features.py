from __future__ import annotations

# 🔥 FIX: Windows → Linux checkpoint compatibility
import sys
import types
import pathlib

fake_module = types.ModuleType("pathlib._local")
fake_module.Path = pathlib.PosixPath
fake_module.PosixPath = pathlib.PosixPath
fake_module.WindowsPath = pathlib.PosixPath
sys.modules["pathlib._local"] = fake_module


import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lfa_stage.lfa_module import LocalFeatureAugmentation
from mtcnn_stage.preprocess import create_mtcnn, process_single_image, resolve_device
from resnet18_stage.raf_db_dataset import IMAGENET_MEAN, IMAGENET_STD, project_root_from_file
from resnet18_stage.train_resnet18 import create_model


# ✅ ADD BACK (fix import error)
def save_heatmap(feature_map: torch.Tensor, heatmap_path: Path, image_size: int) -> None:
    heatmap = feature_map.mean(dim=1).squeeze(0)
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / heatmap.max().clamp(min=1e-6)
    heatmap = (heatmap * 255).byte().cpu().numpy()

    image = Image.fromarray(heatmap, mode="L")
    image = image.resize((image_size, image_size))
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(heatmap_path)


def build_parser() -> argparse.ArgumentParser:
    project_root = project_root_from_file(__file__)
    default_output_dir = project_root / "outputs" / "lfa_outputs"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--margin", type=int, default=20)
    return parser


def load_aligned_tensor(aligned_image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(aligned_image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


def load_backbone(checkpoint: Path | None, device: torch.device) -> torch.nn.Module:
    if checkpoint is None:
        backbone = resnet18(weights=None)
    else:
        full_model = create_model(num_classes=7, weights_mode="none", freeze_backbone=False)

        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)

        full_model.load_state_dict(checkpoint_data["model_state_dict"])
        backbone = full_model

    backbone.eval()
    backbone.to(device)
    return backbone


def forward_to_layer2(backbone: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x = backbone.conv1(inputs)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        x = backbone.layer2(x)
    return x


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    run_name = args.run_name or args.input_image.stem

    aligned_image_path = args.output_dir / f"{run_name}_mtcnn.jpg"
    resnet_map_path = args.output_dir / f"{run_name}_resnet_map.pt"
    lfa_map_path = args.output_dir / f"{run_name}_lfa_map.pt"
    resnet_heatmap_path = args.output_dir / f"{run_name}_resnet_heatmap.jpg"
    lfa_heatmap_path = args.output_dir / f"{run_name}_lfa_heatmap.jpg"

    mtcnn = create_mtcnn(device=device, image_size=args.image_size, margin=args.margin)

    result = process_single_image(
        mtcnn=mtcnn,
        input_path=args.input_image,
        output_path=aligned_image_path,
        copy_if_missed=False,
    )

    if result.status == "error":
        print(result.message)
        return 1

    inputs = load_aligned_tensor(aligned_image_path, args.image_size).to(device)
    backbone = load_backbone(args.checkpoint, device)

    lfa = LocalFeatureAugmentation(channels=128).to(device).eval()

    resnet_feature_map = forward_to_layer2(backbone, inputs)
    with torch.no_grad():
        lfa_feature_map = lfa(resnet_feature_map)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(resnet_feature_map.cpu(), resnet_map_path)
    torch.save(lfa_feature_map.cpu(), lfa_map_path)

    save_heatmap(resnet_feature_map.cpu(), resnet_heatmap_path, args.image_size)
    save_heatmap(lfa_feature_map.cpu(), lfa_heatmap_path, args.image_size)

    print("✅ SUCCESS — Output saved!")
    print(resnet_map_path)
    print(lfa_map_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())