#!/usr/bin/env python3
"""
Prepare YOLOv8 pose labels (single keypoint) from a centers.csv + split txt files.

Expected dataset structure:

dataset_root/
  centers.csv
  splits/
    train.txt
    val.txt
    test.txt

Each split file contains one image path per line (relative paths).
UID is taken from the image filename stem. Example:
  images/train/frame_000123.jpg  -> uid = "frame_000123"

centers.csv must include columns:
  uid, has_ball, x_center, y_center

Output structure (out_root):
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt
  data.yaml
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


REQUIRED_CSV_COLUMNS = {"uid", "has_ball", "x_center", "y_center"}


@dataclass(frozen=True)
class CenterInfo:
    has_ball: int
    x: float
    y: float


def read_split_list(split_txt_path: Path) -> List[str]:
    if not split_txt_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_txt_path}")
    lines = split_txt_path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def uid_from_rel_image_path(rel_path: str) -> str:
    return Path(rel_path).stem


def load_centers_csv(csv_path: Path) -> Dict[str, CenterInfo]:
    if not csv_path.exists():
        raise FileNotFoundError(f"centers.csv not found: {csv_path}")

    data: Dict[str, CenterInfo] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")

        missing = REQUIRED_CSV_COLUMNS.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"centers.csv missing required columns {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            uid = row["uid"].strip()
            if not uid:
                continue

            has_ball = int(row["has_ball"])
            x = float(row["x_center"])
            y = float(row["y_center"])

            data[uid] = CenterInfo(has_ball=has_ball, x=x, y=y)

    return data


def make_pose_label_line(x: float, y: float, box_w: float, box_h: float) -> str:
    """
    YOLOv8 pose format with 1 keypoint:
      class x y w h xk yk vk

    Assumes x,y are normalized in [0, 1].
    vk = 2 means visible keypoint (YOLO pose convention).
    """
    cls = 0
    vk = 2
    return (
        f"{cls} {x:.6f} {y:.6f} {box_w:.6f} {box_h:.6f} "
        f"{x:.6f} {y:.6f} {vk}"
    )


def write_data_yaml(out_root: Path) -> None:
    yaml_text = f"""path: {out_root}
train: images/train
val: images/val
test: images/test

names:
  0: ball

kpt_shape: [1, 3]
"""
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def prepare_yolo_pose_labels_only(
    dataset_root: Path,
    out_root: Path,
    box_w: float = 0.06,
    box_h: float = 0.06,
    log_every: int = 500,
) -> None:
    dataset_root = dataset_root.resolve()
    out_root = out_root.resolve()

    centers = load_centers_csv(dataset_root / "centers.csv")

    splits_dir = dataset_root / "splits"
    split_files = {
        "train": splits_dir / "train.txt",
        "val": splits_dir / "val.txt",
        "test": splits_dir / "test.txt",
    }

    print("\nPreparing YOLOv8 pose labels (1 keypoint)...\n")
    print(f"Dataset root: {dataset_root}")
    print(f"Output root : {out_root}")
    print(f"Box size    : w={box_w}, h={box_h}\n")

    for split, split_file in split_files.items():
        rel_paths = read_split_list(split_file)

        lbl_out_dir = out_root / "labels" / split
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        total = len(rel_paths)
        pos = 0
        neg = 0

        print(f"[{split}] Total images: {total}")

        for i, rel_path in enumerate(rel_paths, start=1):
            uid = uid_from_rel_image_path(rel_path)

            info = centers.get(uid)
            if info is None:
                has_ball, x, y = 0, 0.0, 0.0
            else:
                has_ball, x, y = info.has_ball, info.x, info.y

            lbl_path = lbl_out_dir / f"{uid}.txt"

            if has_ball == 1:
                line = make_pose_label_line(x, y, box_w=box_w, box_h=box_h)
                lbl_path.write_text(line + "\n", encoding="utf-8")
                pos += 1
            else:
                lbl_path.write_text("", encoding="utf-8")
                neg += 1

            if i % log_every == 0 or i == total:
                print(
                    f"  {split}: {i}/{total} "
                    f"({100*i/total:.1f}%) | pos={pos} neg={neg}"
                )

        print(f"[{split} DONE] images={total} positives={pos} negatives={neg}\n")

    write_data_yaml(out_root)

    print("data.yaml written to:")
    print(out_root / "data.yaml")
    print("\nDone. Dataset is ready for YOLOv8 pose training.\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create YOLOv8 pose labels (single keypoint) from centers.csv + splits."
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to dataset root containing centers.csv and splits/.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output dataset root. labels/ and data.yaml will be written here.",
    )
    p.add_argument("--box-w", type=float, default=0.06, help="YOLO box width (normalized).")
    p.add_argument("--box-h", type=float, default=0.06, help="YOLO box height (normalized).")
    p.add_argument("--log-every", type=int, default=500, help="Progress log frequency.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prepare_yolo_pose_labels_only(
        dataset_root=args.dataset_root,
        out_root=args.out_root,
        box_w=args.box_w,
        box_h=args.box_h,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
