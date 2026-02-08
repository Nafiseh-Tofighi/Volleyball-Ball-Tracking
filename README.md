# Volleyball-Ball-Tracking

---

## Prepare your own dataset for Ball Tracking Training

To train the ball tracking model on your own dataset, use:

`tools/prepare_yolo_pose_dataset.py`

This script converts ball center annotations into **YOLOv8 pose format** with one keypoint.

### Dataset Structure

```
dataset_root/
├── centers.csv
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

* Split files contain relative image paths
* Image filename stem is used as the UID

### centers.csv

Required columns:

* `uid`
* `has_ball` (1 or 0)
* `x_center`, `y_center` (normalized)

Images without a ball generate empty label files.

### Output

```
out_root/
├── labels/{train,val,test}
└── data.yaml
```

Label format:
`class x y w h xk yk vk`

### Run

```
python tools/prepare_yolo_pose_dataset.py \
  --dataset-root /path/to/dataset \
  --out-root /path/to/yolo_dataset
```

After this step, the dataset is ready for YOLOv8 pose training.

---

