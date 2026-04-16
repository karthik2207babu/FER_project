from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# reuse existing modules
from lfa_stage.extract_lfa_features import (
    forward_to_layer2,
    load_aligned_tensor,
    load_backbone,
    save_heatmap,
)
from lfa_stage.lfa_module import LocalFeatureAugmentation
from msgc_stage.msgc_module import MultiScaleGlobalConvolution
from safm_stage.safm_module import SpatialAttentionFeatureModule
from mtcnn_stage.preprocess import create_mtcnn, process_single_image, resolve_device
from resnet18_stage.raf_db_dataset import project_root_from_file


def build_parser() -> argparse.ArgumentParser:
    project_root = project_root_from_file(__file__)
    default_output_dir = project_root / "outputs" / "safm_outputs"

    parser = argparse.ArgumentParser(
        description="Run image → MTCNN → ResNet → LFA → MSGC → SAFM"
    )

    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--margin", type=int, default=20)

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

    # output paths
    aligned_image_path = args.output_dir / f"{run_name}_mtcnn.jpg"
    resnet_map_path = args.output_dir / f"{run_name}_resnet_map.pt"
    lfa_map_path = args.output_dir / f"{run_name}_lfa_map.pt"
    msgc_map_path = args.output_dir / f"{run_name}_msgc_map.pt"
    safm_map_path = args.output_dir / f"{run_name}_safm_map.pt"

    resnet_heatmap_path = args.output_dir / f"{run_name}_resnet_heatmap.jpg"
    lfa_heatmap_path = args.output_dir / f"{run_name}_lfa_heatmap.jpg"
    msgc_heatmap_path = args.output_dir / f"{run_name}_msgc_heatmap.jpg"
    safm_heatmap_path = args.output_dir / f"{run_name}_safm_heatmap.jpg"

    # -------- MTCNN --------
    mtcnn = create_mtcnn(device=device, image_size=args.image_size, margin=args.margin)

    mtcnn_result = process_single_image(
        mtcnn=mtcnn,
        input_path=args.input_image,
        output_path=aligned_image_path,
        copy_if_missed=False,
    )

    if mtcnn_result.status == "error":
        print(mtcnn_result.message)
        return 1

    # -------- Load input --------
    inputs = load_aligned_tensor(aligned_image_path, args.image_size).to(device)

    # -------- Backbone --------
    backbone = load_backbone(args.checkpoint, device)

    # -------- Modules --------
    lfa = LocalFeatureAugmentation(channels=128).to(device).eval()
    msgc = MultiScaleGlobalConvolution(channels=128).to(device).eval()
    safm = SpatialAttentionFeatureModule().to(device).eval()

    # -------- Forward pass --------
    resnet_feature_map = forward_to_layer2(backbone, inputs)

    with torch.no_grad():
        lfa_feature_map = lfa(resnet_feature_map)
        msgc_feature_map = msgc(lfa_feature_map)
        safm_feature_map = safm(msgc_feature_map)

    # -------- Move to CPU --------
    resnet_cpu = resnet_feature_map.cpu()
    lfa_cpu = lfa_feature_map.cpu()
    msgc_cpu = msgc_feature_map.cpu()
    safm_cpu = safm_feature_map.cpu()

    # -------- Save --------
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(resnet_cpu, resnet_map_path)
    torch.save(lfa_cpu, lfa_map_path)
    torch.save(msgc_cpu, msgc_map_path)
    torch.save(safm_cpu, safm_map_path)

    save_heatmap(resnet_cpu, resnet_heatmap_path, args.image_size)
    save_heatmap(lfa_cpu, lfa_heatmap_path, args.image_size)
    save_heatmap(msgc_cpu, msgc_heatmap_path, args.image_size)
    save_heatmap(safm_cpu, safm_heatmap_path, args.image_size)

    # -------- Logs --------
    print(f"MTCNN status: {mtcnn_result.status}")
    print(f"ResNet shape: {tuple(resnet_cpu.shape)}")
    print(f"LFA shape: {tuple(lfa_cpu.shape)}")
    print(f"MSGC shape: {tuple(msgc_cpu.shape)}")
    print(f"SAFM shape: {tuple(safm_cpu.shape)}")

    print(f"\nSaved outputs in: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())