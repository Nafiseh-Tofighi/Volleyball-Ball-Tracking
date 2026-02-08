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

## Data Preparation

To train the ball tracking model on a custom dataset, the data must first be converted to
YOLOv8 pose format. This repository provides a preparation script for that purpose.

`tools/prepare_yolo_pose_dataset.py`

### Expected Dataset Structure

```

dataset_root/
├── centers.csv
└── splits/
├── train.txt
├── val.txt
└── test.txt

```

- Split files contain relative image paths
- Image filename stem is used as the unique identifier

### centers.csv

Required columns:

- `uid`
- `has_ball` (1 or 0)
- `x_center`, `y_center` (normalized to [0, 1])

Images without a ball generate empty label files.

### Usage

```

python tools/prepare_yolo_pose_dataset.py 
--dataset-root /path/to/dataset 
--out-root /path/to/yolo_dataset

```

The script generates YOLOv8 pose labels and a `data.yaml` file ready for training.

---

## Training

After preparing the dataset, the model can be trained using the provided training script:

`train.py`

### Usage

```

python train.py --data /path/to/yolo_dataset/data.yaml

```

### Arguments

- `--data` : path to `data.yaml` (required)
- `--model` : base YOLOv8 pose model (default: `yolov8s-pose.pt`)
- `--imgsz` : input image size (default: `640`)
- `--epochs` : number of training epochs (default: `100`)
- `--batch` : batch size (default: `32`)
- `--optimizer` : optimizer type (default: `AdamW`)
- `--lr0` : initial learning rate (default: `1e-3`)
- `--patience` : early stopping patience (default: `20`)
- `--device` : device id or `cpu` (default: `0`)
- `--project` : output directory (default: `runs/pose`)
- `--name` : run name (default: `train`)

Training outputs are saved under `runs/pose/` by default.
```

