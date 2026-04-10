# Computer Vision Final Project

## Introduction

This project implements and compares two object detection models: **YOLOv8** and **SSD300 (VGG16)**.These models detect basketballs and players in basketball game footage. Both models are trained on the same annotated dataset and evaluated using standard object detection metrics (mAP, precision, recall). 

## File Contents

### `archive/`

Contains the basketball detection dataset, sourced from [Roboflow](https://app.roboflow.com). The dataset includes **3,591 images** annotated in YOLOv8 format with two classes: **BasketBall** and **Player**. The folder is organized into three splits:

- `train/` — Training images and labels
- `valid/` — Validation images and labels
- `test/` — Test images and labels
- `data.yaml` — Dataset configuration file defining class names, paths, and split locations

### `runs/`

Contains the output artifacts generated during YOLO model training and evaluation. This includes:

- `detect/yolo_runs/basketball_detect/` — Full training run output: training/validation batch visualizations, performance curves (F1, Precision, Recall, PR), confusion matrices, `results.csv` with per-epoch metrics, and saved model weights (`best.pt`, `last.pt`)
- `detect/val/` and `detect/val2/` — Validation run outputs with prediction visualizations, performance curves, and confusion matrices

### `YOLO_Basketball_Detection.ipynb`

Jupyter notebook that handles the full YOLOv8 pipeline:

- Loads and configures the YOLOv8n (nano) model for fine-tuning on the basketball dataset
- Trains the model on the `archive/` dataset
- Evaluates the trained model on the test set, computing mAP@50 (0.838), precision (0.824), and recall (0.794)
- Saves evaluation metrics to `yolo_metrics.json`

### `SSD_Basketball_Detection.ipynb`

Jupyter notebook that handles the full SSD300 pipeline:

- Builds an SSD300 model with a VGG16 backbone, modifying the classification head for 3 classes (background + BasketBall + Player)
- Trains the model on the same dataset with custom data loaders and training loops
- Evaluates the trained model on the test set, computing mAP@50 (0.823), precision (0.710), and recall (0.944)
- Saves the best model weights to `ssd_best.pth` and metrics to `ssd_metrics.json`
