"""
Visual inspection on a subset of test images.

Draws:
- GT square in green (if GT has ball)
- Prediction square in red (if prediction exists)

Prints per image:
- TP TN FP FN case
- pixel error for TP

Optionally saves a grid image if --save-dir is provided.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize YOLOv8 pose predictions on sample test images")
    p.add_argument("--weights", type=Path, required=True, help="Path to model weights, for example best.pt")
    p.add_argument("--centers-csv", type=Path, required=True, help="Path to centers.csv (GT source)")
    p.add_argument("--images", type=Path, required=True, help="Path to test images directory")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--n", type=int, default=25, help="Number of images to show")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--half-box", type=int, default=10, help="Half size of drawn square in pixels")
    return p.parse_args()


def draw_square(img_rgb: np.ndarray, x: float, y: float, half_box: int, color_rgb: tuple[int, int, int]) -> None:
    h, w = img_rgb.shape[:2]
    x1 = int(max(0, x - half_box))
    y1 = int(max(0, y - half_box))
    x2 = int(min(w - 1, x + half_box))
    y2 = int(min(h - 1, y + half_box))
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color_rgb, 2)


def main() -> None:
    args = parse_args()

    model = YOLO(str(args.weights))

    centers_df = pd.read_csv(args.centers_csv)
    if "uid" not in centers_df.columns:
        raise ValueError("centers.csv must contain a 'uid' column")
    centers_df = centers_df.set_index("uid")

    test_imgs = sorted(args.images.glob("*.png"))
    if not test_imgs:
        test_imgs = sorted(args.images.glob("*.jpg"))
    if not test_imgs:
        test_imgs = sorted(args.images.glob("*.jpeg"))
    if not test_imgs:
        raise FileNotFoundError(f"No images found in {args.images}")

    rng = random.Random(args.seed)
    rng.shuffle(test_imgs)
    test_imgs = test_imgs[: min(args.n, len(test_imgs))]

    tp = tn = fp = fn = 0
    pixel_errors: list[float] = []

    print("\nPer image results:\n")

    for idx, img_path in enumerate(test_imgs, start=1):
        uid = img_path.stem

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[{idx:02d}] {uid} | failed to read image")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        gt_has = 0
        gt_x = gt_y = None
        if uid in centers_df.index:
            row = centers_df.loc[uid]
            if int(row.get("has_ball", 0)) == 1:
                gt_has = 1
                gt_x = float(row["x_center"]) * w
                gt_y = float(row["y_center"]) * h

        r = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

        pred_has = 0
        pred_x = pred_y = None
        if r.keypoints is not None and len(r.keypoints) > 0:
            pred_has = 1
            px, py = r.keypoints.xy[0, 0]
            pred_x = float(px)
            pred_y = float(py)

        if gt_has == 1 and pred_has == 1:
            case = "TP"
            tp += 1
        elif gt_has == 0 and pred_has == 0:
            case = "TN"
            tn += 1
        elif gt_has == 1 and pred_has == 0:
            case = "FN"
            fn += 1
        else:
            case = "FP"
            fp += 1

        pix_err = None
        norm_err = None
        if case == "TP" and gt_x is not None and gt_y is not None and pred_x is not None and pred_y is not None:
            pix_err = float(np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2))
            norm_err = pix_err / float(np.sqrt(w * w + h * h))
            pixel_errors.append(pix_err)

        if gt_has == 1 and gt_x is not None and gt_y is not None:
            draw_square(img_rgb, gt_x, gt_y, args.half_box, (0, 255, 0))

        if pred_has == 1 and pred_x is not None and pred_y is not None:
            draw_square(img_rgb, pred_x, pred_y, args.half_box, (255, 0, 0))

        gt_str = "None" if gt_has == 0 else f"({gt_x:.1f},{gt_y:.1f})"
        pr_str = "None" if pred_has == 0 else f"({pred_x:.1f},{pred_y:.1f})"
        err_str = "-" if pix_err is None else f"{pix_err:.2f}px ({norm_err:.4f})"

        print(
            f"[{idx:02d}] {uid} | GT={gt_has} Pred={pred_has} | case={case} | "
            f"GTxy={gt_str} Predxy={pr_str} | err={err_str}"
        )

        title = f"{uid} | {case} | GT={gt_has} Pred={pred_has}"
        if pix_err is not None:
            title += f" | err={pix_err:.1f}px"

        plt.figure(figsize=(7, 5))
        plt.title(title)
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.show()

    print("\nSummary on shown samples:")
    print(f"  TP: {tp}")
    print(f"  TN: {tn}")
    print(f"  FP: {fp}")
    print(f"  FN: {fn}")

    if pixel_errors:
        pe = np.array(pixel_errors, dtype=float)
        print("\nCenter error on TP samples:")
        print(f"  mean   : {pe.mean():.2f} px")
        print(f"  median : {np.median(pe):.2f} px")
        print(f"  90%    : {np.percentile(pe, 90):.2f} px")
    else:
        print("\nNo TP samples in the shown set, cannot compute center error.")


if __name__ == "__main__":
    main()
