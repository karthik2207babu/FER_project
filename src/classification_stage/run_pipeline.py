from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import your master pipeline from train.py!
from train import FERFullPipeline
from lfa_stage.extract_lfa_features import load_aligned_tensor
from mtcnn_stage.preprocess import create_mtcnn, process_single_image, resolve_device

EMOTIONS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

def build_parser() -> argparse.ArgumentParser:
    # Use a safe default output directory
    default_output_dir = Path("outputs/final_outputs")

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

    # 1. Run MTCNN to crop the face
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

    # 2. Load the cropped image
    inputs = load_aligned_tensor(aligned_image_path, args.image_size).to(device)

    # 3. Initialize the Master Pipeline and load your trained weights
    model = FERFullPipeline().to(device)
    if args.checkpoint is not None and args.checkpoint.exists():
        print(f"🧠 Loading trained brain from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    else:
        print("⚠️ Warning: No checkpoint found. Using random weights!")
        
    model.eval()

    # 4. The Forward Pass
    with torch.no_grad():
        logits = model(inputs)
        
        # Convert logits to percentages
        probs = F.softmax(logits, dim=1).squeeze(0)
        predicted_idx = torch.argmax(probs).item()

    # 5. Print Results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- FINAL PREDICTION ---")
    
    for i, prob in enumerate(probs):
        print(f"{EMOTIONS[i]:>10}: {prob.item() * 100:.2f}%")
        
    print(f"\n=> Predicted Emotion: {EMOTIONS[predicted_idx]}")

    return 0

if __name__ == "__main__":
    sys.exit(main())