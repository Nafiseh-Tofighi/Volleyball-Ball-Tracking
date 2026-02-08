#!/usr/bin/env python3
"""
Numeric evaluation on the full test set.

Computes presence metrics:
- TP TN FP FN
- Accuracy Precision Recall F1

Computes localization error (TP only):
- mean median 75% 90%

Optionally saves per image results to a CSV file.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate YOLOv8 pose model on full test set")
    p.add_argument("--weights", type=Path, required=True, help="Path to model weights, for example best.pt")
    p.add_argument("--centers-csv", type=Path, required=True, help="Path to centers.csv (GT source)")
    p.add_argument("--images", type=Path, required=True, help="Path to test images directory")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--split", type=str, default="test", help="Split name in centers.csv, if split column exists")
    p.add_argument("--save-csv", type=Path, default=None, help="Optional path to save per image CSV report")
    return p.parse_args()


def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def main() -> None:
    args = parse_args()

    model = YOLO(str(args.weights))

    centers_df = pd.read_csv(args.centers_csv)
    if "uid" not in centers_df.columns:
        raise ValueError("centers.csv must contain a 'uid' column")

    if "split" in centers_df.columns:
        centers_df = centers_df[centers_df["split"] == args.split]

    centers_df = centers_df.set_index("uid")
    if len(centers_df) == 0:
        raise ValueError("No samples found after applying split filtering")

    img_paths = sorted(list(args.images.glob("*.png")) + list(args.images.glob("*.jpg")) + list(args.images.glob("*.jpeg")))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {args.images}")

    tp = tn = fp = fn = 0
    pixel_errors: list[float] = []
    rows: list[dict] = []

    print("\nRunning numeric evaluation on the full test set...\n")

    for img_path in tqdm(img_paths):
        uid = img_path.stem
        if uid not in centers_df.index:
            continue

        row = centers_df.loc[uid]
        gt_has = int(row.get("has_ball", 0))

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_x = gt_y = None
        if gt_has == 1:
            gt_x = float(row["x_center"]) * w
            gt_y = float(row["y_center"]) * h

        r = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

        if r.keypoints is not None and len(r.keypoints) > 0:
            pred_has = 1
            px, py = r.keypoints.xy[0, 0]
            pred_x = float(px)
            pred_y = float(py)
        else:
            pred_has = 0
            pred_x = pred_y = None

        if gt_has == 1 and pred_has == 1:
            tp += 1
            case = "TP"
            err = float(np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2))
            pixel_errors.append(err)
        elif gt_has == 0 and pred_has == 0:
            tn += 1
            case = "TN"
            err = None
        elif gt_has == 1 and pred_has == 0:
            fn += 1
            case = "FN"
            err = None
        else:
            fp += 1
            case = "FP"
            err = None

        rows.append(
            {
                "uid": uid,
                "GT_has_ball": gt_has,
                "Pred_has_ball": pred_has,
                "case": case,
                "pixel_error": err,
            }
        )

    total = tp + tn + fp + fn
    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    print("\n=== Presence metrics ===")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 score : {f1:.4f}")

    if pixel_errors:
        pe = np.array(pixel_errors, dtype=float)
        print("\n=== Center localization error (TP only, pixels) ===")
        print(f"Mean   : {pe.mean():.2f}")
        print(f"Median : {np.median(pe):.2f}")
        print(f"75%    : {np.percentile(pe, 75):.2f}")
        print(f"90%    : {np.percentile(pe, 90):.2f}")
    else:
        print("\nNo true positives found, cannot compute localization error.")

    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.save_csv, index=False)
        print(f"\nPer image report saved to:\n{args.save_csv}")


if __name__ == "__main__":
    main()
