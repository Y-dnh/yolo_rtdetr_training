# YOLO / RT-DETR — IR Detection Training Pipeline

Pipeline для навчання, валідації та експорту моделей детекції на інфрачервоних (тепловізійних) зображеннях.
Підтримує архітектури **YOLO** та **RT-DETR** через уніфікований API Ultralytics.

Класи: custom

---

## Як працює пайплайн навчання (train.py)

Покроковий потік від конфігів до виклику `model.train()`.

### Крок 1. Константи на початку файлу

У `train.py` оголошуються три блоки:

| Константа | Призначення |
|-----------|-------------|
| **TRAINING_CONFIG** | Параметри тренування: `data`, `epochs`, `batch`, `imgsz`, `optimizer`, `lr0`, ваги втрат (`box`, `cls`, `dfl`) тощо. Без аугментацій. |
| **AUGMENTATION_CONFIG** | Вбудовані аугментації YOLO/RT-DETR: `hsv_h`, `hsv_v`, `degrees`, `translate`, `scale`, `mosaic`, `erasing` тощо. |
| **CUSTOM_TRANSFORMS** | Список об'єктів Albumentations (CLAHE, GaussNoise, RandomFog, PixelDropout тощо) — кастомний пайплайн для IR. |

Ключі всередині словників у lowercase — це імена аргументів API Ultralytics, їх змінювати не можна.

### Крок 2. Запуск: main() → train_model()

- `main()` встановлює seed, перевіряє середовище, викликає `train_model(**training_kwargs)`.

### Крок 3. Збір конфігу в train_model()

1. **Виклик `get_train_config(**kwargs)`**
   - Формується один словник: `config = {**TRAINING_CONFIG, **AUGMENTATION_CONFIG, **kwargs}`.
   - Застосовується **фільтр по MODEL_TYPE**: для RT-DETR видаляються YOLO-only ключі (`dfl`, `nbs`, `mosaic`, `close_mosaic` тощо); для YOLO — RT-DETR-only (поки порожній набір).
   - Саме тут **AUGMENTATION_CONFIG** потрапляє в конфіг.

2. **Дописування Albumentations**
   - `config["augmentations"] = CUSTOM_TRANSFORMS` — у конфіг додається глобальний список трансформ. Ultralytics зберігає його у `args.yaml` у теці run.

### Крок 4. Завантаження моделі та навчання

- **load_model(model_path)** — для `.yaml` (наприклад P2) при увімкненому `YAML_TRANSFER_LEARNING` завантажуються базові ваги відповідного `.pt`; шари з невідповідними розмірами пропускаються.
- **model.train(**config)** — один виклик з усіма параметрами: тренувальні налаштування, вбудовані аугментації та `augmentations=CUSTOM_TRANSFORMS`.

Якщо Ultralytics не приймає параметр `augmentations` (наприклад у майбутніх версіях RT-DETR), викидається зрозуміла помилка з вказівкою параметра.

### Крок 5. Після навчання

- Ваги зберігаються у `runs/<project>/<name>/train/weights/best.pt` та `last.pt`.
- Автоматично викликається **export_model_after_training()** — експорт у ONNX з `EXPORT_CONFIG` (з фільтром по MODEL_TYPE для ключів на кшталт `nms`).

### Підсумок потоку

```
TRAINING_CONFIG + AUGMENTATION_CONFIG + kwargs
        → get_train_config() (фільтр по MODEL_TYPE)
        → config["augmentations"] = CUSTOM_TRANSFORMS
        → load_model(path)
        → model.train(**config)
        → export_model_after_training(best.pt)
```

---

## Підтримувані архітектури

| Архітектура | Тип | NMS | Перемикач |
|-------------|-----|-----|-----------|
| **YOLO** (v8, v11, тощо) | Anchor-free CNN | Так (post-process) | `MODEL_TYPE = "yolo"` |
| **RT-DETR** (L, X) | Transformer (end-to-end) | Ні (NMS-free) | `MODEL_TYPE = "rtdetr"` |

Переключення між архітектурами — зміна однієї змінної `MODEL_TYPE` на початку кожного скрипта.
Конфіг фільтрується автоматично: YOLO-only параметри видаляються при використанні RT-DETR і навпаки.

## Структура проекту

Усі запуски (тренування та валідація) зберігаються в папці **`runs/`**. Під нею — назва проєкту, далі — теки конкретних експериментів (train, val тощо).

```
yolo_training/
├── train.py                 # Навчання + автоматичний ONNX експорт
├── validate.py              # Валідація + звіти (JSON, Markdown)
├── export.py                # Експорт моделі (ONNX, TensorRT, OpenVINO, тощо)
├── coco_to_yolo.py          # Конвертер COCO -> YOLO формат
├── prepare_dataset.py       # Розбивка датасету на train/val/test
├── requirements.txt
├── README.md
├── dataset_split/           # Підготовлений датасет (опційно)
│   ├── train/images/, train/labels/
│   ├── val/, test/
│   └── data.yaml
└── runs/                    # Усі результати (тренування + валідація)
    └── <PROJECT_NAME>/      # Назва проєкту (наприклад yolo26m, yolo26s_p2)
        ├── baseline/        # Експеримент навчання (TRAINING_CONFIG["name"])
        │   ├── train/       # Перший run (train2, train3… при повторних запусках)
        │   │   ├── weights/
        │   │   │   ├── best.pt, last.pt
        │   │   │   └── best.onnx
        │   │   ├── args.yaml
        │   │   ├── results.csv, results.png
        │   │   └── …
        │   └── train2/      # (якщо запуск повторено без exist_ok)
        └── validation_26s_yaml/  # Експеримент валідації (validate.py EXPERIMENT_NAME)
            ├── val/
            │   ├── predictions.json
            │   ├── validation_results.json
            │   └── …
            └── val2/
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

### Запуск у WSL (Linux)

Проєкт можна повністю використовувати з WSL2: ті самі скрипти, конфіги та датасет на диску `D:` (доступний як `/mnt/d/` — копіювати нічого не потрібно).

1. **Шлях до даних**  
   У `train.py` і `validate.py` шлях до датасету береться з змінної середовища `YOLO_DATASET_ROOT`. Якщо вона не задана, використовується `D:/dataset_for_training` (Windows).

2. **Формат `data.yaml` датасету (обов'язково для WSL)**  
   У файлі **`D:\dataset_for_training\data.yaml`** мають бути **відносні** шляхи, а не абсолютні Windows-шляхи. Інакше в WSL виникне помилка на кшталт `missing path '.../d:\dataset_for_training\valid\images'`.

   **Правильно** (працює і в Windows, і в WSL):
   ```yaml
   path: .
   train: train/images
   val: valid/images
   nc: 3
   names: ['person', 'car', 'truck']
   ```

   **Неправильно:** `path: D:/dataset_for_training` або `train: D:/dataset_for_training/train/images` — такі шляхи в WSL дають змішані шляхи і помилку.

3. **У WSL один раз у сесії (або в `~/.bashrc`):**
   ```bash
   export YOLO_DATASET_ROOT=/mnt/d/dataset_for_training
   ```

4. **Створити env і встановити залежності (у WSL):**
   ```bash
   cd /mnt/d/projects_yaroslav/yolo_training
   conda create -n yolo_training_env python=3.11 -y
   conda activate yolo_training_env
   pip install -r requirements.txt
   ```

5. **Запуск тренування:**
   ```bash
   export YOLO_DATASET_ROOT=/mnt/d/dataset_for_training
   python train.py
   ```
   Валідація: той самий `export`, далі `python validate.py` (при потребі зміни `PROJECT_NAME` та шлях до ваг у `validate.py`).

6. **GPU у WSL2:** потрібен драйвер NVIDIA у Windows та [CUDA Toolkit для WSL](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSLUbuntu). Після цього PyTorch побачить GPU.

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

Конфігурація — на початку `train.py` у трьох блоках:

```python
MODEL_TYPE = "yolo"                    # або "rtdetr"
PRETRAINED_MODEL = "yolo26m.pt"       # або "rtdetr-x.pt"

# 1) Параметри тренування (без аугментацій)
TRAINING_CONFIG = {
    "epochs": 50,
    "batch": 16,
    "imgsz": 1024,
    "optimizer": "SGD",
    "lr0": 0.01,
    "cos_lr": True,
    # ...
}

# 2) Вбудовані аугментації YOLO/RT-DETR (додаються в get_train_config)
AUGMENTATION_CONFIG = {
    "hsv_v": 0.6, "degrees": 5.0, "translate": 0.2, "scale": 0.25,
    "mosaic": 1.0, "fliplr": 0.5, "erasing": 0.0,
    # ...
}

# 3) Кастомний пайплайн Albumentations (передається як augmentations=...)
CUSTOM_TRANSFORMS = [
    A.CLAHE(...), A.RandomGamma(...), A.GaussNoise(...),
    # ...
]
```

Після навчання автоматично експортує модель в ONNX.

**Результати:**
- `<PROJECT_NAME>/baseline/weights/best.pt` — найкраща модель
- `<PROJECT_NAME>/baseline/weights/best.onnx` — ONNX експорт
- `<PROJECT_NAME>/baseline/args.yaml` — конфіг запуску з усіма параметрами (Ultralytics зберігає також `augmentations`)
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
