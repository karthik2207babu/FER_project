from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lfa_stage.extract_lfa_features import (
    forward_to_layer2,
    load_aligned_tensor,
    load_backbone,
    save_heatmap,
)
from lfa_stage.lfa_module import LocalFeatureAugmentation
from msgc_stage.msgc_module import MultiScaleGlobalConvolution
from mtcnn_stage.preprocess import create_mtcnn, process_single_image, resolve_device
from resnet18_stage.raf_db_dataset import project_root_from_file


def build_parser() -> argparse.ArgumentParser:
    project_root = project_root_from_file(__file__)
    default_output_dir = project_root / "outputs" / "msgc_outputs"

    parser = argparse.ArgumentParser(
        description=(
            "Run image -> MTCNN -> ResNet-18(layer2) -> LFA -> MSGC and save the "
            "multi-scale feature map."
        )
    )
    parser.add_argument("--input-image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where intermediate maps and MSGC output will be saved.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional file prefix.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional ResNet-18 checkpoint. Only backbone layers up to layer2 are used.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use GPU automatically if available, or force CPU/CUDA.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Input size after MTCNN.")
    parser.add_argument("--margin", type=int, default=20, help="Face crop margin for MTCNN.")
    return parser


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
    resnet_map_path = args.output_dir / f"{run_name}_resnet_map.pt"
    lfa_map_path = args.output_dir / f"{run_name}_lfa_map.pt"
    msgc_map_path = args.output_dir / f"{run_name}_msgc_map.pt"
    resnet_heatmap_path = args.output_dir / f"{run_name}_resnet_heatmap.jpg"
    lfa_heatmap_path = args.output_dir / f"{run_name}_lfa_heatmap.jpg"
    msgc_heatmap_path = args.output_dir / f"{run_name}_msgc_heatmap.jpg"

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

    inputs = load_aligned_tensor(aligned_image_path, args.image_size).to(device)
    backbone = load_backbone(args.checkpoint, device)
    lfa = LocalFeatureAugmentation(channels=128).to(device)
    msgc = MultiScaleGlobalConvolution(channels=128).to(device)

    resnet_feature_map = forward_to_layer2(backbone, inputs)
    with torch.no_grad():
        lfa_feature_map = lfa(resnet_feature_map)
        msgc_feature_map = msgc(lfa_feature_map)

    resnet_feature_map_cpu = resnet_feature_map.cpu()
    lfa_feature_map_cpu = lfa_feature_map.cpu()
    msgc_feature_map_cpu = msgc_feature_map.cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(resnet_feature_map_cpu, resnet_map_path)
    torch.save(lfa_feature_map_cpu, lfa_map_path)
    torch.save(msgc_feature_map_cpu, msgc_map_path)
    save_heatmap(resnet_feature_map_cpu, resnet_heatmap_path, args.image_size)
    save_heatmap(lfa_feature_map_cpu, lfa_heatmap_path, args.image_size)
    save_heatmap(msgc_feature_map_cpu, msgc_heatmap_path, args.image_size)

    print(f"MTCNN status: {mtcnn_result.status}")
    print(f"Aligned image saved to: {aligned_image_path}")
    print(f"ResNet feature map shape: {tuple(resnet_feature_map_cpu.shape)}")
    print(f"LFA feature map shape: {tuple(lfa_feature_map_cpu.shape)}")
    print(f"MSGC feature map shape: {tuple(msgc_feature_map_cpu.shape)}")
    print(f"ResNet feature map saved to: {resnet_map_path}")
    print(f"LFA feature map saved to: {lfa_map_path}")
    print(f"MSGC feature map saved to: {msgc_map_path}")
    print(f"ResNet heatmap saved to: {resnet_heatmap_path}")
    print(f"LFA heatmap saved to: {lfa_heatmap_path}")
    print(f"MSGC heatmap saved to: {msgc_heatmap_path}")
    if args.checkpoint is not None:
        print(f"Checkpoint used: {args.checkpoint}")
        print("Note: ResNet is initialized from the checkpoint; LFA and MSGC still need end-to-end training.")
    else:
        print("Checkpoint used: none (untrained ResNet backbone, LFA, and MSGC).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
