"""
Модуль для навчання YOLO / RT-DETR моделі детекції на IR (інфрачервоних) зображеннях.
Класи: person, car, truck
Усі параметри конфігурації знаходяться на початку файлу.

Перемикач MODEL_TYPE дозволяє обрати архітектуру:
  - "yolo"   -> ultralytics.YOLO
  - "rtdetr"  -> ultralytics.RTDETR
"""

import os
import sys
import re

# Фікс для правильного відображення tqdm у Windows PowerShell
if sys.platform == 'win32':
    os.system('')  # Включає ANSI escape sequences підтримку
    # Альтернативно можна вимкнути кольори якщо не допомагає:
    # os.environ['NO_COLOR'] = '1'

import random
import numpy as np
import torch
from ultralytics import YOLO, RTDETR
import ultralytics


# =============================================================================
# ВИБІР АРХІТЕКТУРИ: "yolo" або "rtdetr"
# =============================================================================
VALID_MODEL_TYPES = {"yolo", "rtdetr"}
MODEL_TYPE = "yolo"        # <-- ПЕРЕМИКАЧ: "yolo" або "rtdetr"

# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ
# =============================================================================
SEED = 42

# --- YOLO конфіг ---
PROJECT_NAME = "yolo11x_for_autolabelling"
PRETRAINED_MODEL = "D:/projects_yaroslav/yolo_training/yolo11x_for_autolabelling/baseline/weights/last.pt"

# --- Transfer Learning для YAML моделей (P2/P6 та інші кастомні архітектури) ---
# Якщо True і model_path є .yaml файл — автоматично завантажить базові ваги (.pt)
# Наприклад: yolov8s-p2.yaml → завантажить ваги з yolov8s.pt
#            yolo11m-p2.yaml → завантажить ваги з yolo11m.pt
#            yolo26-p6.yaml  → завантажить ваги з yolo26.pt
# Шари з невідповідними розмірами (detection heads) будуть пропущені автоматично.
YAML_TRANSFER_LEARNING = True

# --- RT-DETR конфіг (розкоментувати при MODEL_TYPE = "rtdetr") ---
# PROJECT_NAME = "rtdetr-x_for_autolabelling"
# PRETRAINED_MODEL = "rtdetr-x.pt"  # rtdetr-l.pt або rtdetr-x.pt
# Шляхи до даних
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_ROOT = os.path.join(BASE_DIR, "dataset_split")
DATASET_ROOT = "D:/dataset_for_training"
PROJECT_DIR = os.path.join(BASE_DIR, PROJECT_NAME)
YAML_PATH = os.path.join(DATASET_ROOT, "data.yaml")


# =============================================================================
# ПАРАМЕТРИ, СПЕЦИФІЧНІ ДЛЯ КОЖНОЇ АРХІТЕКТУРИ
# Ці множини використовуються для автоматичної фільтрації конфігу
# =============================================================================

# Ключі, які є ТІЛЬКИ у YOLO (будуть видалені при MODEL_TYPE="rtdetr")
YOLO_ONLY_TRAIN_KEYS = {
    "dfl",              # Distribution Focal Loss — тільки YOLO
    "nbs",              # Nominal batch size
    "overlap_mask",     # Segmentation mask overlap
    "mask_ratio",       # Segmentation mask ratio
    "pose",             # Pose estimation loss weight
    "kobj",             # Keypoint objectness loss weight
    "close_mosaic",     # Epoch to disable mosaic — YOLO mosaic pipeline
    "mosaic",           # Mosaic augmentation probability
    "copy_paste",       # Copy-paste augmentation
    "copy_paste_mode",  # Copy-paste mode
    "multi_scale",      # Multi-scale training
}

# Ключі, які є ТІЛЬКИ у RT-DETR (будуть видалені при MODEL_TYPE="yolo")
RTDETR_ONLY_TRAIN_KEYS: set[str] = set()  # Поки немає унікальних — ultralytics приймає спільні

# Ключі експорту, які специфічні для YOLO
YOLO_ONLY_EXPORT_KEYS = {
    "nms",              # RT-DETR — NMS-free архітектура, nms завжди має бути False
}


# ============================================================================
# ПАРАМЕТРИ НАВЧАННЯ (передаються як **kwargs до model.train())
# =============================================================================
TRAINING_CONFIG = {
    # Модель та дані
    "data": YAML_PATH,
    "project": PROJECT_DIR,
    "name": "baseline",
    "exist_ok": False,

    # ==========================================================================
    # ЗАГАЛЬНІ ПАРАМЕТРИ НАВЧАННЯ ДЛЯ IR (ТЕПЛОВІЗІЙНИХ) ЗОБРАЖЕНЬ
    # ==========================================================================
    "epochs": 100,          # Більше епох для кращої збіжності на IR даних
    "time": None,
    "patience": 20,        # Більше терпіння - IR дані можуть потребувати більше часу
    "batch": 4,            # Менший batch для великої моделі
    "imgsz": 1024,         # Стандартний розмір
    "save": True,
    "save_period": -1,
    "cache": False,
    "device": 0,
    "workers": 8,
    "seed": SEED,
    "deterministic": True,
    "single_cls": False,
    "classes": None,
    "rect": False,          # Rectangular training — зберігає aspect ratio
    "multi_scale": False,   # [YOLO-only] Multi-scale для кращої генералізації
    "cos_lr": False,        # Cosine LR scheduler — плавніше навчання. RT-DETR рекомендовано: True
    "close_mosaic": 10,     # [YOLO-only] Вимкнути mosaic пізніше для fine-tuning
    "resume": True,
    "amp": False,
    "fraction": 1.0,
    "profile": False,
    "freeze": None,
    "val": True,
    "plots": True,
    "compile": False,

    # ==========================================================================
    # ОПТИМІЗАТОР ТА LEARNING RATE ДЛЯ IR ЗОБРАЖЕНЬ
    # ==========================================================================
    "pretrained": False,
    "optimizer": "AdamW",
    "lr0": 0.001,             # RT-DETR рекомендовано: 0.0001 (трансформер потребує нижчий LR)
    "lrf": 0.01,
    "momentum": 0.937,        # AdamW зазвичай має beta1=0.937
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,    # RT-DETR рекомендовано: 5.0 (більше warmup для трансформера)
    "warmup_momentum": 0.5,
    "warmup_bias_lr": 0.01,

    # ==========================================================================
    # ВАГИ ФУНКЦІЙ ВТРАТ ДЛЯ IR ДЕТЕКЦІЇ
    # YOLO: box + cls + dfl
    # RT-DETR: Hungarian matching + GIOU + L1 + CE (dfl/nbs/overlap_mask/... ігноруються)
    # ==========================================================================
    "box": 7.5,             # Вага box loss (спільний)
    "cls": 1.0,             # Збільшено — важливо розрізняти ...
    "dfl": 1.5,             # [YOLO-only] Distribution Focal Loss
    "pose": 12.0,           # [YOLO-only] Pose estimation loss weight
    "kobj": 1.0,            # [YOLO-only] Keypoint objectness
    "nbs": 64,              # [YOLO-only] Nominal batch size
    "overlap_mask": True,   # [YOLO-only] Mask overlap
    "mask_ratio": 4,        # [YOLO-only] Mask ratio
    "dropout": 0.0,
    "label_smoothing": 0.0,

    # ==========================================================================
    # АУГМЕНТАЦІЯ ДЛЯ ІНФРАЧЕРВОНИХ (ТЕПЛОВІЗІЙНИХ) ЗОБРАЖЕНЬ
    # ==========================================================================
    # HSV аугментації вимкнені — зображення вже grayscale
    "hsv_h": 0.0,           # Вимкнено — немає кольору
    "hsv_s": 0.0,           # Вимкнено — немає насиченості
    "hsv_v": 0.3,           # Невелика варіація яскравості

    # Геометричні трансформації
    "degrees": 5.0,         # Мінімальний поворот — камера на мачті стабільна
    "translate": 0.15,      # Зсув зображення
    "scale": 0.0,           # Масштабування (об'єкти на різних відстанях)
    "shear": 2.0,           # Невеликий зсув перспективи
    "perspective": 0.0,     # Вимкнено — фіксована висота камери

    # Відображення
    "flipud": 0.0,          # Вимкнено — камера завжди зверху
    "fliplr": 0.5,          # Горизонтальне відображення
    "bgr": 0.0,             # Вимкнено — grayscale

    # Композитні аугментації
    "mosaic": 0.0,          # [YOLO-only] Mosaic аугментація
    "mixup": 0.0,           # Невеликий mixup для регуляризації
    "cutmix": 0.0,          # Вимкнено
    "copy_paste": 0.0,      # [YOLO-only] Copy-paste для малих об'єктів (person)
    "copy_paste_mode": "flip",  # [YOLO-only]

    # Інші аугментації
    "auto_augment": "",     # Вимкнено — стандартні аугментації не для IR
    "erasing": 0.3,         # Random erasing для регуляризації
}


# =============================================================================
# КОНФІГУРАЦІЯ ЕКСПОРТУ ONNX (після навчання)
# Підтримувані параметри для ONNX: imgsz, half, dynamic, simplify, opset, nms, batch, device
# Документація: https://docs.ultralytics.com/modes/export/#arguments
# =============================================================================
EXPORT_CONFIG = {
    "format": "onnx",
    "imgsz": 1024,              # Розмір входу (має відповідати imgsz з навчання)
    "half": True,               # FP16 — зменшує розмір, прискорює інференс на GPU
    "dynamic": False,           # Динамічний розмір входу при інференсі
    "simplify": True,           # Спрощення графу через onnxslim
    "opset": 12,                # ONNX opset версія (None = остання, 11-13 для сумісності)
    "nms": False,               # [YOLO-only] Вбудувати NMS в модель (RT-DETR: завжди False)
    "batch": 1,                 # Batch size
    "device": 0,                # None = авто, 0 = GPU, "cpu" = CPU
}


def validate_model_type() -> None:
    """Перевірка що MODEL_TYPE має допустиме значення."""
    if MODEL_TYPE not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Невідомий MODEL_TYPE: '{MODEL_TYPE}'. "
            f"Допустимі значення: {sorted(VALID_MODEL_TYPES)}"
        )


def _derive_base_weights(yaml_name: str) -> str:
    """
    Визначає назву файлу базових ваг із назви YAML-файлу.
    Видаляє відомі суфікси архітектури/задачі і замінює .yaml на .pt.
    
    Підтримувані суфікси (та їх комбінації):
        -p2, -p6          — додаткові detection heads
        -obb              — Oriented Bounding Boxes
        -seg              — Instance Segmentation
        -pose             — Pose Estimation
        -cls              — Classification
        -world, -worldv2  — YOLO-World
    
    Приклади:
        yolov8s-p2.yaml   → yolov8s.pt
        yolov8x-p2.yaml   → yolov8x.pt
        yolo11m-p2.yaml   → yolo11m.pt
        yolo26-p6.yaml    → yolo26.pt
        yolo26s-obb.yaml  → yolo26s.pt
        yolo26x-obb.yaml  → yolo26x.pt
        yolov8n-seg.yaml  → yolov8n.pt
        yolov8s-pose.yaml → yolov8s.pt
        yolov8s-p2-seg.yaml → yolov8s.pt
        yolov8s-worldv2.yaml → yolov8s.pt
    
    Args:
        yaml_name: Назва YAML файлу (без шляху)
    
    Returns:
        Назва файлу базових ваг (.pt)
    """
    # Видаляємо розширення .yaml
    stem = yaml_name.replace(".yaml", "")
    # Видаляємо відомі суфікси: -p2, -p6, -obb, -seg, -pose, -cls, -world, -worldv2
    # Підтримуються комбінації: -p2-seg, -p2-obb тощо
    base_stem = re.sub(r"(?:-(?:p\d+|obb|seg|pose|cls|world(?:v\d+)?))+$", "", stem)
    return f"{base_stem}.pt"


def load_model(model_path: str):
    """
    Завантаження моделі відповідно до MODEL_TYPE.
    
    Transfer Learning (YAML_TRANSFER_LEARNING=True):
        Якщо model_path є .yaml файл — створює архітектуру з YAML,
        автоматично визначає базові ваги та завантажує їх через model.load().
        Шари з невідповідними розмірами (detection heads P2/P6) пропускаються.
    
    Args:
        model_path: Шлях до моделі або назва архітектури (.pt або .yaml)
    
    Returns:
        Завантажена модель (YOLO або RTDETR)
    
    Raises:
        ValueError: Якщо MODEL_TYPE невідомий
    """
    validate_model_type()

    # RT-DETR — завжди стандартне завантаження
    if MODEL_TYPE == "rtdetr" or "rtdetr" in model_path.lower():
        print(f"[Model] Завантаження RT-DETR: {model_path}")
        return RTDETR(model_path)

    # YOLO з YAML + Transfer Learning
    if model_path.lower().endswith(".yaml") and YAML_TRANSFER_LEARNING:
        yaml_name = os.path.basename(model_path)
        base_weights = _derive_base_weights(yaml_name)

        print(f"[Model] Transfer Learning режим:")
        print(f"  Архітектура (YAML): {model_path}")
        print(f"  Базові ваги:        {base_weights}")

        # 1. Ініціалізуємо структуру моделі з YAML
        model = YOLO(model_path)

        # 2. Завантажуємо базові ваги (matching шари)
        #    model.load() автоматично пропускає шари з невідповідними розмірами
        #    і виводить WARNING для кожного пропущеного шару
        try:
            model.load(base_weights)
            print(f"[Model] ✅ Базові ваги '{base_weights}' завантажено успішно!")
            print(f"[Model]    Шари з невідповідними розмірами (heads) пропущені автоматично.")
        except FileNotFoundError:
            print(f"[Model] ⚠️ Файл ваг '{base_weights}' не знайдено!")
            print(f"[Model]    Модель буде навчатися з нуля (random init).")
        except Exception as e:
            print(f"[Model] ⚠️ Помилка при завантаженні ваг '{base_weights}': {e}")
            print(f"[Model]    Модель буде навчатися з нуля (random init).")

        return model

    # Стандартне завантаження .pt або .yaml без transfer learning
    print(f"[Model] Завантаження YOLO: {model_path}")
    return YOLO(model_path)


def filter_config(config: dict, excluded_keys: set) -> dict:
    """
    Фільтрує конфігурацію: видаляє ключі, несумісні з поточною архітектурою.
    
    Args:
        config: Вхідний словник конфігурації
        excluded_keys: Множина ключів, які потрібно видалити
    
    Returns:
        dict: Відфільтрований словник конфігурації
    """
    removed = set(config.keys()) & excluded_keys
    if removed:
        print(f"[Config] MODEL_TYPE='{MODEL_TYPE}' -> видалено несумісні ключі: {sorted(removed)}")

    return {k: v for k, v in config.items() if k not in excluded_keys}


def get_train_config(**kwargs) -> dict:
    """
    Повертає відфільтрований training config для поточного MODEL_TYPE.
    
    Args:
        **kwargs: Параметри, що перезаписують TRAINING_CONFIG
    
    Returns:
        dict: Готовий конфіг для model.train()
    """
    config = {**TRAINING_CONFIG, **kwargs}

    if MODEL_TYPE == "rtdetr":
        return filter_config(config, YOLO_ONLY_TRAIN_KEYS)
    elif MODEL_TYPE == "yolo":
        return filter_config(config, RTDETR_ONLY_TRAIN_KEYS)
    return config


def get_export_config(**kwargs) -> dict:
    """
    Повертає відфільтрований export config для поточного MODEL_TYPE.
    
    Args:
        **kwargs: Параметри, що перезаписують EXPORT_CONFIG
    
    Returns:
        dict: Готовий конфіг для model.export()
    """
    config = {**EXPORT_CONFIG, **kwargs}

    if MODEL_TYPE == "rtdetr":
        return filter_config(config, YOLO_ONLY_EXPORT_KEYS)
    elif MODEL_TYPE == "yolo":
        return filter_config(config, set())
    return config


def setup_seed(seed: int) -> None:
    """Ініціалізація seed для відтворюваності результатів."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_environment() -> str:
    """Налаштування середовища та створення директорій."""
    ultralytics.checks()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(PROJECT_DIR, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Project Directory: {PROJECT_DIR}")
    
    return device


def train_model(
    model_path: str = PRETRAINED_MODEL,
    **kwargs
) -> tuple:
    """
    Навчання YOLO моделі.
    
    Args:
        model_path: Шлях до попередньо навченої моделі
        **kwargs: Параметри навчання (перезаписують TRAINING_CONFIG)
    
    Returns:
        tuple: (results, trained_model_path)
    """
    # Отримуємо відфільтрований конфіг для поточного MODEL_TYPE
    config = get_train_config(**kwargs)
    
    print(f"Завантаження моделі: {model_path}")
    model = load_model(model_path)
    
    print("Початок навчання...")
    print(f"Архітектура: {MODEL_TYPE.upper()}")
    print(f"Модель: {model_path}")
    print(f"Конфігурація: epochs={config['epochs']}, batch={config['batch']}, imgsz={config['imgsz']}")
    print(f"Оптимізатор: {config['optimizer']}, lr0={config['lr0']}, cos_lr={config['cos_lr']}")
    
    # Навчання моделі
    results = model.train(**config)
    
    # Шлях до найкращої моделі
    trained_model_path = os.path.join(
        config["project"],
        config["name"],
        "weights",
        "best.pt"
    )
    
    print("Навчання завершено!")
    print(f"Найкраща модель збережена: {trained_model_path}")
    
    # Експорт моделі в ONNX
    onnx_path = export_model_after_training(trained_model_path)
        
    return results, trained_model_path, onnx_path


def export_model_after_training(
    model_path: str,
    export_config: dict = None
) -> str | None:
    """
    Експорт моделі після навчання.
    
    Args:
        model_path: Шлях до навченої моделі (.pt)
        export_config: Конфігурація експорту (за замовчуванням EXPORT_CONFIG)
    
    Returns:
        str | None: Шлях до експортованої моделі або None при помилці
    """
    if export_config is None:
        export_config = EXPORT_CONFIG.copy()
    
    # Фільтруємо за архітектурою та прибираємо None значення
    config = get_export_config(**export_config)
    config = {k: v for k, v in config.items() if v is not None}
    
    print("\n" + "=" * 60)
    print("ЕКСПОРТ МОДЕЛІ")
    print("=" * 60)
    print(f"Модель: {model_path}")
    print(f"Формат: {config.get('format', 'onnx')}")
    print(f"Розмір зображення: {config.get('imgsz', 640)}")
    print(f"Динамічний вхід: {config.get('dynamic', False)}")
    print(f"FP16: {config.get('half', False)}")
    print(f"INT8: {config.get('int8', False)}")
    print("=" * 60)
    
    try:
        model = load_model(model_path)
        exported_path = model.export(**config)
        print(f"\nЕкспорт завершено успішно!")
        print(f"Файл збережено: {exported_path}")
        return exported_path
    except Exception as e:
        print(f"\nПомилка під час експорту: {e}")
        return None


def main():
    """Головна функція для запуску навчання."""
    # Налаштування seed
    setup_seed(SEED)
    
    # Налаштування середовища
    device = setup_environment()
    
    # Оновлення device в конфігурації якщо потрібно
    training_kwargs = {"device": 0 if device == "cuda" else "cpu"}
    
    # Запуск навчання
    results, model_path, onnx_path = train_model(**training_kwargs)
    
    return results, model_path, onnx_path


if __name__ == "__main__":
    main()
