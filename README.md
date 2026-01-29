# YOLO IR Detection Training Pipeline

A complete pipeline for training YOLO models. This project includes tools for COCO to YOLO format conversion, dataset preparation, and model training/validation.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Convert COCO to YOLO Format](#1-convert-coco-to-yolo-format)
  - [2. Prepare Dataset](#2-prepare-dataset)
  - [3. Train Model](#3-train-model)
  - [4. Validate Model](#4-validate-model)
- [Configuration](#configuration)
- [Dataset Structure](#dataset-structure)
- [Training Parameters](#training-parameters)
- [Thermal Image Considerations](#thermal-image-considerations)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## ✨ Features

- **COCO to YOLO Conversion**: Convert COCO format annotations to YOLO format with support for:
  - Bounding boxes
  - Segmentation masks (optional)
  - Keypoints (optional)
  - Class mapping (91 to 80 classes)

- **Dataset Preparation**: Automated splitting into train/val/test sets with:
  - Configurable split ratios
  - Automatic YAML configuration generation
  - Label verification

- **Validation Tools**: Model validation and performance evaluation

## 📁 Project Structure

```
project/
├── coco_to_yolo.py          # COCO format converter
├── prepare_dataset.py       # Dataset splitting tool
├── train.py                 # Training script
├── validate.py              # Validation script
├── requirements.txt         # Python dependencies
├── dataset/                 # Source data (create this)
│   ├── images/
│   └── labels/
├── dataset_split/           # Prepared dataset (auto-generated)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── yolo.yaml
└── yolov8x-p2_for_autolabelling/  # Training outputs (auto-generated)
    └── baseline/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        └── results.png
```

## Installation

### Setup

1. Clone or download this project:
```bash
git clone <url>
cd yolo_training
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📖 Usage

### 1. Convert COCO to YOLO Format

If your annotations are in COCO format, convert them first:

```python
python coco_to_yolo.py
```

**Configuration** (edit in `coco_to_yolo.py`):
```python
CONFIG = {
    "labels_dir": "path/to/coco/annotations",  # Folder with JSON files
    "json_file": None,                         # Specific JSON file (or None for all)
    "save_dir": "path/to/output/labels",       # Output folder
    "use_segments": False,                     # Include segmentation masks
    "use_keypoints": False,                    # Include keypoints
    "cls91to80": False,                        # Map 91 COCO classes to 80
    "class_offset": 0,                         # Class ID offset (1 for COCO)
}
```

### 2. Prepare Dataset

Split your dataset into train/val/test sets:

```python
python prepare_dataset.py
```

**Configuration** (edit in `prepare_dataset.py`):
```python
# Split ratios
TRAIN_RATIO = ...
VAL_RATIO = ...
TEST_RATIO = ...

# Source paths
SOURCE_IMAGES = Path("dataset/images")
SOURCE_LABELS = Path("dataset/labels")

# Classes (adjust for your dataset)
CLASSES = {...}
```

The script will:
- Split images randomly with a fixed seed
- Copy images and labels to respective folders
- Generate `yolo.yaml` configuration
- Report statistics

### 3. Train Model

```python
python train.py
```

**Key Configuration** (edit in `train.py`):

Training outputs:
- Best model: `PROJECT_NAME/baseline/weights/best.pt`
- Last model: `PROJECT_NAME/baseline/weights/last.pt`
- Training plots: `PROJECT_NAME/baseline/results.png`
- Logs: `PROJECT_NAME/baseline/`

### 4. Validate Model

Evaluate model performance:

```python
python validate.py
```

**Configuration** (edit in `validate.py`):

