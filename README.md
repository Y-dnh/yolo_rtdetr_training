# YOLO / RT-DETR — IR Detection Training Pipeline

Pipeline для навчання, валідації та експорту моделей детекції на інфрачервоних (тепловізійних) зображеннях.
Підтримує архітектури **YOLO** та **RT-DETR** через уніфікований API Ultralytics.

Класи: custom

## Підтримувані архітектури

| Архітектура | Тип | NMS | Перемикач |
|-------------|-----|-----|-----------|
| **YOLO** (v8, v11, тощо) | Anchor-free CNN | Так (post-process) | `MODEL_TYPE = "yolo"` |
| **RT-DETR** (L, X) | Transformer (end-to-end) | Ні (NMS-free) | `MODEL_TYPE = "rtdetr"` |

Переключення між архітектурами — зміна однієї змінної `MODEL_TYPE` на початку кожного скрипта.
Конфіг фільтрується автоматично: YOLO-only параметри видаляються при використанні RT-DETR і навпаки.

## Структура проекту

```
yolo_rtdetr_training/
├── train.py                 # Навчання + автоматичний ONNX експорт
├── validate.py              # Валідація на test split + звіти (JSON, Markdown)
├── export.py                # Експорт моделі (ONNX, TensorRT, OpenVINO, тощо)
├── coco_to_yolo.py          # Конвертер COCO -> YOLO формат
├── prepare_dataset.py       # Розбивка датасету на train/val/test
├── requirements.txt         # Залежності
├── README.md
├── dataset_split/           # Підготовлений датасет (auto-generated)
│   ├── train/images/, train/labels/
│   ├── val/images/, val/labels/
│   ├── test/images/, test/labels/
│   └── data.yaml
└── <PROJECT_NAME>/          # Результати навчання (auto-generated)
    └── baseline/
        ├── weights/
        │   ├── best.pt
        │   ├── best.onnx
        │   └── last.pt
        ├── args.yaml        # Повний конфіг навчання (від ultralytics)
        ├── results.csv      # Метрики по епохах
        └── results.png
```

## Встановлення

```bash
# 1. Клонувати
git clone <url>
cd yolo_rtdetr_training

# 2. Віртуальне середовище
python -m venv yolo_rtdetr_training_env
yolo_rtdetr_training_env\Scripts\activate       # Windows
# source yolo_rtdetr_training_env/bin/activate  # Linux/Mac

# 3. Залежності
pip install -r requirements.txt

# 4. Перевірка
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Використання

### Перемикач архітектури

На початку кожного скрипта (`train.py`, `validate.py`, `export.py`) є:

```python
MODEL_TYPE = "yolo"        # <-- ПЕРЕМИКАЧ: "yolo" або "rtdetr"
```

При невірному значенні (наприклад `"rsdetr"`, `"tolo"`) — скрипт кидає `ValueError` з переліком допустимих значень.

Параметри, помічені `[YOLO-only]` в конфігу, автоматично видаляються при `MODEL_TYPE = "rtdetr"`. В консолі буде лог:
```
[Config] MODEL_TYPE='rtdetr' -> видалено несумісні ключі: ['close_mosaic', 'copy_paste', 'dfl', ...]
```

### 1. Підготовка датасету

```bash
# Конвертація COCO -> YOLO (якщо потрібно)
python coco_to_yolo.py

# Розбивка на train/val/test
python prepare_dataset.py
```

### 2. Навчання

```bash
python train.py
```

Конфігурація — на початку `train.py`:

```python
MODEL_TYPE = "yolo"                    # або "rtdetr"
PRETRAINED_MODEL = "yolo11x.pt"        # або "rtdetr-x.pt"

TRAINING_CONFIG = {
    "epochs": 50,
    "batch": 4,
    "imgsz": 1024,
    "optimizer": "AdamW",
    "lr0": 0.001,          # RT-DETR рекомендовано: 0.0001
    "warmup_epochs": 3.0,  # RT-DETR рекомендовано: 5.0
    "cos_lr": False,       # RT-DETR рекомендовано: True
    # ...
}
```

Після навчання автоматично експортує модель в ONNX.

**Результати:**
- `<PROJECT_NAME>/baseline/weights/best.pt` — найкраща модель
- `<PROJECT_NAME>/baseline/weights/best.onnx` — ONNX експорт
- `<PROJECT_NAME>/baseline/args.yaml` — повний конфіг (логування ultralytics)
- `<PROJECT_NAME>/baseline/results.csv` — метрики по епохах

### 3. Валідація

```bash
python validate.py
```

Конфігурація — на початку `validate.py`:

```python
MODEL_TYPE = "yolo"                    # або "rtdetr"

VALIDATION_CONFIG = {
    "conf": 0.5,
    "iou": 0.5,
    "imgsz": 960,
    "split": "test",
    # ...
}
```

**Результати:**
- `validation_results.json` — метрики в JSON
- `validation_report.md` — Markdown звіт з таблицями
- Консольний вивід з mAP, Precision, Recall, F1, FPS

### 4. Експорт

```bash
python export.py
```

Конфігурація — на початку `export.py`:

```python
MODEL_TYPE = "yolo"                    # або "rtdetr"

EXPORT_CONFIG = {
    "format": "onnx",
    "imgsz": (540, 960),
    "half": False,
    "dynamic": True,
    "simplify": True,
    # ...
}
```

**Підтримувані формати:**

| Формат | Аргумент | Призначення |
|--------|----------|-------------|
| ONNX | `onnx` | Універсальний, CPU/GPU |
| TensorRT | `engine` | NVIDIA GPU (до 5x прискорення) |
| OpenVINO | `openvino` | Intel CPU/GPU |
| TFLite | `tflite` | Мобільні/Edge пристрої |
| CoreML | `coreml` | Apple (macOS/iOS) |
| NCNN | `ncnn` | ARM (мобільні/embedded) |
| TorchScript | `torchscript` | PyTorch deployment |

## YOLO vs RT-DETR — різниця в конфігурації

### Параметри, що автоматично фільтруються

При `MODEL_TYPE = "rtdetr"` з конфігу навчання видаляються:

| Параметр | Причина |
|----------|---------|
| `dfl` | Distribution Focal Loss — тільки YOLO |
| `nbs` | Nominal batch size — YOLO-specific |
| `close_mosaic` | Mosaic pipeline — тільки YOLO |
| `mosaic` | Mosaic аугментація — YOLO-specific |
| `copy_paste`, `copy_paste_mode` | Copy-paste аугментація — YOLO |
| `multi_scale` | Multi-scale training — YOLO |
| `overlap_mask`, `mask_ratio` | Segmentation mask — YOLO |
| `pose`, `kobj` | Pose/Keypoint loss — YOLO |

При експорті: `nms` видаляється для RT-DETR (NMS-free архітектура).

### Рекомендовані значення для RT-DETR

Трансформерна архітектура потребує інших гіперпараметрів. Рекомендації вказані коментарями біля відповідних параметрів в `train.py`:

| Параметр | YOLO | RT-DETR |
|----------|------|---------|
| `lr0` | 0.001 | 0.0001 |
| `warmup_epochs` | 3.0 | 5.0 |
| `cos_lr` | False | True |

### Loss функції

- **YOLO**: `box` + `cls` + `dfl` (Distribution Focal Loss)
- **RT-DETR**: Hungarian matching + GIOU + L1 + Cross-Entropy (параметри `box` та `cls` спільні)
