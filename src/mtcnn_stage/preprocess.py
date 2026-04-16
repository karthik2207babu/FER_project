from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import torch
    from facenet_pytorch import MTCNN
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:
    missing_name = getattr(exc, "name", "required package")
    raise SystemExit(
        f"Missing dependency: {missing_name}. "
        "Install the project requirements first, then rerun this script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "RAF-DB"
DEFAULT_DATASET_DIR = DEFAULT_DATA_ROOT / "DATASET"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "MTCNN"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class ImageResult:
    status: str
    confidence: float | None = None
    message: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MTCNN on a single image or preprocess the RAF-DB dataset into "
            "a new folder without touching the original files."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("single", "dataset"),
        default="single",
        help="Use 'single' for one image or 'dataset' for RAF-DB preprocessing.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use GPU if available with 'auto', or force CPU/CUDA explicitly.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Final face crop size saved by MTCNN.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=40,
        help="Extra pixels kept around the detected face.",
    )

    parser.add_argument(
        "--input-image",
        type=Path,
        default=PROJECT_ROOT / "test_images" / "test.jpg",
        help="Image used in single-image mode.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=PROJECT_ROOT / "test_images" / "output.jpg",
        help="Saved crop path in single-image mode.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="RAF-DB root that contains DATASET/, train_labels.csv, and test_labels.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where processed images and copied label CSVs will be saved.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="both",
        help="Which RAF-DB split to preprocess in dataset mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on processed samples per split for quick testing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already exist inside the output folder.",
    )
    copy_group = parser.add_mutually_exclusive_group()
    copy_group.add_argument(
        "--copy-if-missed",
        dest="copy_if_missed",
        action="store_true",
        help="Copy the original image when MTCNN cannot detect a face.",
    )
    copy_group.add_argument(
        "--drop-if-missed",
        dest="copy_if_missed",
        action="store_false",
        help="Do not save anything when MTCNN misses a face.",
    )
    parser.set_defaults(copy_if_missed=True)
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_mtcnn(device: torch.device, image_size: int, margin: int) -> MTCNN:
    return MTCNN(
        image_size=image_size,
        margin=margin,
        select_largest=True,
        post_process=False,
        device=device,
    )


def score_face(mtcnn: MTCNN, image: Image.Image) -> float | None:
    boxes, probabilities = mtcnn.detect(image)
    if boxes is None or probabilities is None or len(probabilities) == 0:
        return None
    return float(max(probabilities))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def process_single_image(
    mtcnn: MTCNN,
    input_path: Path,
    output_path: Path,
    copy_if_missed: bool,
) -> ImageResult:
    if not input_path.exists():
        return ImageResult(status="error", message=f"Input image not found: {input_path}")
    if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return ImageResult(
            status="error",
            message=f"Unsupported image type: {input_path.suffix}",
        )

    try:
        with Image.open(input_path) as image_file:
            image = image_file.convert("RGB")
            confidence = score_face(mtcnn, image)
            ensure_parent(output_path)
            face = mtcnn(image, save_path=str(output_path))
    except UnidentifiedImageError:
        return ImageResult(status="error", message=f"Unreadable image file: {input_path}")
    except OSError as exc:
        return ImageResult(status="error", message=f"Failed to read {input_path}: {exc}")

    if face is not None:
        return ImageResult(status="detected", confidence=confidence)

    if copy_if_missed:
        shutil.copy2(input_path, output_path)
        return ImageResult(
            status="copied",
            confidence=confidence,
            message="Face not detected. Original image copied instead.",
        )

    return ImageResult(status="missed", confidence=confidence, message="No face detected.")


def iter_label_rows(labels_csv: Path) -> Iterable[tuple[str, str]]:
    with labels_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            image_name = (row.get("image") or "").strip()
            label = (row.get("label") or "").strip()
            if image_name and label:
                yield image_name, label


def copy_label_file(data_root: Path, output_root: Path, split: str) -> None:
    src = data_root / f"{split}_labels.csv"
    dst = output_root / f"{split}_labels.csv"
    if src.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def process_split(
    mtcnn: MTCNN,
    data_root: Path,
    output_root: Path,
    split: str,
    limit: int | None,
    skip_existing: bool,
    copy_if_missed: bool,
) -> dict[str, int]:
    dataset_dir = data_root / "DATASET" / split
    labels_csv = data_root / f"{split}_labels.csv"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {dataset_dir}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {labels_csv}")

    stats = {
        "total": 0,
        "detected": 0,
        "copied": 0,
        "missed": 0,
        "skipped": 0,
        "errors": 0,
    }

    start_time = time.time()
    for image_name, label in iter_label_rows(labels_csv):
        if limit is not None and stats["total"] >= limit:
            break

        input_path = dataset_dir / label / image_name
        output_path = output_root / split / label / image_name

        if skip_existing and output_path.exists():
            stats["skipped"] += 1
            continue

        stats["total"] += 1
        result = process_single_image(
            mtcnn=mtcnn,
            input_path=input_path,
            output_path=output_path,
            copy_if_missed=copy_if_missed,
        )

        if result.status in stats:
            stats[result.status] += 1
        else:
            stats["errors"] += 1
            print(f"[{split}] {result.message}")

        if stats["total"] % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"[{split}] processed {stats['total']} images "
                f"in {elapsed:.1f}s"
            )

    copy_label_file(data_root, output_root, split)
    return stats


def run_single_mode(mtcnn: MTCNN, args: argparse.Namespace) -> int:
    result = process_single_image(
        mtcnn=mtcnn,
        input_path=args.input_image,
        output_path=args.output_image,
        copy_if_missed=args.copy_if_missed,
    )
    if result.status == "error":
        print(result.message)
        return 1

    print(f"Status: {result.status}")
    if result.confidence is not None:
        print(f"Detection confidence: {result.confidence:.4f}")
    print(f"Saved to: {args.output_image}")
    if result.message:
        print(result.message)
    return 0


def run_dataset_mode(mtcnn: MTCNN, args: argparse.Namespace) -> int:
    splits = ("train", "test") if args.split == "both" else (args.split,)
    overall_exit_code = 0

    for split in splits:
        print(f"\nProcessing split: {split}")
        try:
            stats = process_split(
                mtcnn=mtcnn,
                data_root=args.data_root,
                output_root=args.output_root,
                split=split,
                limit=args.limit,
                skip_existing=args.skip_existing,
                copy_if_missed=args.copy_if_missed,
            )
        except FileNotFoundError as exc:
            print(str(exc))
            overall_exit_code = 1
            continue

        print(
            f"[{split}] total={stats['total']} detected={stats['detected']} "
            f"copied={stats['copied']} missed={stats['missed']} "
            f"skipped={stats['skipped']} errors={stats['errors']}"
        )

    print(f"\nProcessed data saved under: {args.output_root}")
    return overall_exit_code


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.input_image = args.input_image.resolve()
    args.output_image = args.output_image.resolve()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    mtcnn = create_mtcnn(
        device=device,
        image_size=args.image_size,
        margin=args.margin,
    )

    if args.mode == "dataset":
        return run_dataset_mode(mtcnn, args)
    return run_single_mode(mtcnn, args)


if __name__ == "__main__":
    sys.exit(main())
