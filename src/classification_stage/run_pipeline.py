from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# imports
from lfa_stage.extract_lfa_features import (
    forward_to_layer2,
    load_aligned_tensor,
    load_backbone,
)
from lfa_stage.lfa_module import LocalFeatureAugmentation
from msgc_stage.msgc_module import MultiScaleGlobalConvolution
from safm_stage.safm_module import SpatialAttentionFeatureModule
from tokenization_stage.tokenization_module import RegionTokenization
from frit_stage.frit_module import FRITTransformer
from classification_stage.classification_module import EmotionClassifier
from mtcnn_stage.preprocess import create_mtcnn, process_single_image, resolve_device
from resnet18_stage.raf_db_dataset import project_root_from_file

EMOTIONS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

def build_parser() -> argparse.ArgumentParser:
    project_root = project_root_from_file(__file__)
    default_output_dir = project_root / "outputs" / "final_outputs"

    parser = argparse.ArgumentParser()
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
    aligned_image_path = args.output_dir / f"{run_name}_mtcnn.jpg"

    # mtcnn
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

    # load
    inputs = load_aligned_tensor(aligned_image_path, args.image_size).to(device)
    backbone = load_backbone(args.checkpoint, device)

    # modules
    lfa = LocalFeatureAugmentation(channels=128).to(device).eval()
    msgc = MultiScaleGlobalConvolution(channels=128).to(device).eval()
    safm = SpatialAttentionFeatureModule().to(device).eval()
    tokenization = RegionTokenization().to(device).eval()
    frit = FRITTransformer().to(device).eval()
    classifier = EmotionClassifier().to(device).eval()

    # forward
    resnet_feature_map = forward_to_layer2(backbone, inputs)

    with torch.no_grad():
        lfa_map = lfa(resnet_feature_map)
        msgc_map = msgc(lfa_map)
        safm_map = safm(msgc_map)
        tokens = tokenization(safm_map)
        frit_out = frit(tokens)
        logits = classifier(frit_out)
        
        # probabilities
        probs = F.softmax(logits, dim=1).squeeze(0)
        predicted_idx = torch.argmax(probs).item()

    # save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # logs
    print(f"\n--- FINAL PREDICTION ---")
    print(f"Logits shape: {tuple(logits.shape)}")
    
    for i, prob in enumerate(probs):
        print(f"{EMOTIONS[i]:>10}: {prob.item() * 100:.2f}%")
        
    print(f"\n=> Predicted Emotion: {EMOTIONS[predicted_idx]}")

    return 0

if __name__ == "__main__":
    sys.exit(main())