from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtcnn_stage.preprocess import create_mtcnn, process_single_image, resolve_device
from resnet18_stage.raf_db_dataset import IMAGENET_MEAN, IMAGENET_STD, project_root_from_file
from resnet18_stage.train_resnet18 import create_model


def build_parser() -> argparse.ArgumentParser:
    project_root = project_root_from_file(__file__)
    default_output_dir = project_root / "outputs" / "feature_maps"

    parser = argparse.ArgumentParser(
        description="Run MTCNN and then stop at the ResNet-18 feature map."
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where aligned image and extracted features will be saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom prefix for saved files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional trained ResNet-18 checkpoint (.pt). If omitted, an untrained ResNet-18 is used.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use GPU automatically if available, or force CPU/CUDA.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input size for both MTCNN output and ResNet-18 input.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=20,
        help="Extra pixels around the face when MTCNN crops the image.",
    )
    return parser


def load_aligned_tensor(aligned_image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(aligned_image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


def load_model(checkpoint: Path | None, device: torch.device) -> torch.nn.Module:
    model = create_model(num_classes=7, weights_mode="none", freeze_backbone=False)

    if checkpoint is not None:
        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data["model_state_dict"])

    model.eval()
    model.to(device)
    return model


def forward_to_feature_map(model: torch.nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        x = model.conv1(inputs)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        feature_map = model.layer2(x)

        pooled = model.avgpool(feature_map)
        feature_vector = torch.flatten(pooled, 1)

    return feature_map, feature_vector


def save_heatmap(feature_map: torch.Tensor, heatmap_path: Path) -> None:
    heatmap = feature_map.mean(dim=1).squeeze(0)
    heatmap = heatmap - heatmap.min()
    denominator = heatmap.max().clamp(min=1e-6)
    heatmap = heatmap / denominator
    heatmap = (heatmap * 255).byte().cpu().numpy()

    image = Image.fromarray(heatmap, mode="L")
    image = image.resize((224, 224))
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(heatmap_path)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.input_image = args.input_image.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    run_name = args.run_name or args.input_image.stem
    aligned_image_path = args.output_dir / f"{run_name}_mtcnn.jpg"
    feature_map_path = args.output_dir / f"{run_name}_feature_map.pt"
    feature_vector_path = args.output_dir / f"{run_name}_feature_vector.pt"
    heatmap_path = args.output_dir / f"{run_name}_feature_heatmap.jpg"

    mtcnn = create_mtcnn(device=device, image_size=args.image_size, margin=args.margin)
    mtcnn_result = process_single_image(
        mtcnn=mtcnn,
        input_path=args.input_image,
        output_path=aligned_image_path,
        copy_if_missed=True,
    )
    if mtcnn_result.status == "error":
        print(mtcnn_result.message)
        return 1

    model = load_model(checkpoint=args.checkpoint, device=device)
    inputs = load_aligned_tensor(aligned_image_path, args.image_size).to(device)
    feature_map, feature_vector = forward_to_feature_map(model, inputs)

    feature_map_cpu = feature_map.cpu()
    feature_vector_cpu = feature_vector.cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(feature_map_cpu, feature_map_path)
    torch.save(feature_vector_cpu, feature_vector_path)
    save_heatmap(feature_map_cpu, heatmap_path)

    print(f"MTCNN status: {mtcnn_result.status}")
    print(f"Aligned image saved to: {aligned_image_path}")
    print(f"Feature map shape: {tuple(feature_map_cpu.shape)}")
    print(f"Feature vector shape: {tuple(feature_vector_cpu.shape)}")
    print(f"Feature map tensor saved to: {feature_map_path}")
    print(f"Feature vector tensor saved to: {feature_vector_path}")
    print(f"Feature heatmap saved to: {heatmap_path}")
    if args.checkpoint is not None:
        print(f"Checkpoint used: {args.checkpoint}")
    else:
        print("Checkpoint used: none (untrained ResNet-18 backbone)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
