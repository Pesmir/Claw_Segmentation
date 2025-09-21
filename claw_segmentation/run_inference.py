"""Command-line entry point that executes the claw segmentation inference pipeline."""
import argparse
from pathlib import Path

try:  # Allow running as `python claw_segmentation/run_inference.py`
    from .inference_pipeline import InferenceConfig, run_inference
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys

    package_root = Path(__file__).resolve().parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from inference_pipeline import InferenceConfig, run_inference


DEFAULT_CONFIG = InferenceConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run claw segmentation inference")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_CONFIG.model_path,
        help="Path to the YOLO model weights",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_CONFIG.data_dir,
        help="Root directory containing dated image folders",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_CONFIG.result_dir,
        help="Directory where inference artifacts will be written",
    )
    parser.add_argument(
        "--excel-filename",
        type=str,
        default=DEFAULT_CONFIG.excel_filename,
        help="Excel file (relative to data dir) providing group/category assignments",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIG.conf,
        help="Confidence threshold forwarded to YOLO",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_CONFIG.iou,
        help="IoU threshold forwarded to YOLO",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=DEFAULT_CONFIG.max_det,
        help="Maximum number of detections per image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InferenceConfig(
        model_path=args.model_path,
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        excel_filename=args.excel_filename,
    )
    final_df = run_inference(config)
    print(f"Processed {len(final_df)} rows. Updated results stored in {config.result_dir}.")


if __name__ == "__main__":
    main()
