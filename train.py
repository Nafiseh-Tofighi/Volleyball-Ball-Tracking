#!/usr/bin/env python3
"""
Train YOLOv8 pose model for ball tracking.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 pose model for ball tracking"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to data.yaml generated during dataset preparation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-pose.pt",
        help="Base YOLOv8 pose model",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--project",
        type=Path,
        default="runs/pose",
        help="Directory to save training outputs",
    )
    parser.add_argument("--name", type=str, default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)

    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        verbose=True,
    )


if __name__ == "__main__":
    main()
