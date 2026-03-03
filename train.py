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
import tempfile
try:
    import yaml
except ModuleNotFoundError:
    from yaml_shim import yaml  # fallback якщо PyYAML зламаний (наприклад Windows wheel 6.x)

# Фікс для правильного відображення tqdm у Windows PowerShell
if sys.platform == 'win32':
    os.system('')  # Включає ANSI escape sequences підтримку
    # Альтернативно можна вимкнути кольори якщо не допомагає:
    # os.environ['NO_COLOR'] = '1'

import random
import numpy as np
import torch
import albumentations as A
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
PROJECT_NAME = "yolo26m"
PRETRAINED_MODEL = "yolo26m.pt"

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
# Шляхи до даних: все під runs/<назва проєкту>/ (тут тренування та валідації)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
# DATASET_ROOT = os.path.join(BASE_DIR, "dataset_split")
# У WSL задай: export YOLO_DATASET_ROOT=/mnt/d/dataset_for_training
DATASET_ROOT = os.environ.get("YOLO_DATASET_ROOT", "D:/dataset_for_training")
PROJECT_DIR = os.path.join(RUNS_DIR, PROJECT_NAME)
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
# ПАРАМЕТРИ НАВЧАННЯ (без аугментацій; аугментації нижче)
# У get_train_config() додаються **AUGMENTATION_CONFIG → фільтр по MODEL_TYPE.
# У train_model() додається config["augmentations"] = CUSTOM_TRANSFORMS → model.train(**config).
# ============================================================================
TRAINING_CONFIG = {
    "data": YAML_PATH,
    "project": PROJECT_DIR,
    "name": "baseline",
    "exist_ok": False,

    # ==========================================================================
    # ЗАГАЛЬНІ ПАРАМЕТРИ НАВЧАННЯ ДЛЯ IR (ТЕПЛОВІЗІЙНИХ) ЗОБРАЖЕНЬ
    # ==========================================================================
    "epochs": 50,          # Більше епох для кращої збіжності на IR даних
    "time": None,
    "patience": 10,
    "batch": 8,
    "imgsz": 1024,
    "save": True,
    "save_period": -1,
    "cache": False,
    "device": 0,
    "workers": 12,
    "seed": SEED,
    "deterministic": True,
    "single_cls": False,
    "classes": None,
    "rect": False,          # Rectangular training — зберігає aspect ratio
    "multi_scale": False,   # [YOLO-only] Multi-scale для кращої генералізації
    "cos_lr": True,         # Плавне косинусне згасання LR (обов'язково для IR)
    "close_mosaic": 10,     # [YOLO-only] Вимкнути mosaic пізніше для fine-tuning
    "resume": False,
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
    "pretrained": True,
    "optimizer": "SGD",       # SGD для глибшого та плавнішого пошуку мінімуму
    "lr0": 0.005,             # Стартовий LR для SGD
    "lrf": 0.01,              # Кінцевий LR
    "momentum": 0.937,
    "weight_decay": 0.001,    # Посилений штраф для боротьби з оверфітом
    "warmup_epochs": 5.0,     # RT-DETR рекомендовано: 5.0 (більше warmup для трансформера)
    "warmup_momentum": 0.5,
    "warmup_bias_lr": 0.01,

    # ==========================================================================
    # ВАГИ ФУНКЦІЙ ВТРАТ ДЛЯ IR ДЕТЕКЦІЇ
    # YOLO: box + cls + dfl
    # RT-DETR: Hungarian matching + GIOU + L1 + CE (dfl/nbs/overlap_mask/... ігноруються)
    # ==========================================================================
    "box": 10.0,            # Пріоритет на точність рамок (мікро-об'єкти)
    "cls": 1.0,
    "dfl": 2.0,             # [YOLO-only] Ідеальне облягання країв об'єктів
    "pose": 12.0,           # [YOLO-only] Pose estimation loss weight
    "kobj": 1.0,            # [YOLO-only] Keypoint objectness
    "nbs": 64,              # [YOLO-only] Nominal batch size
    "overlap_mask": True,   # [YOLO-only] Mask overlap
    "mask_ratio": 4,        # [YOLO-only] Mask ratio
    "dropout": 0.0,
    "label_smoothing": 0.0,
}

# =============================================================================
# АУГМЕНТАЦІЯ (вбудована YOLO/RT-DETR). Додається в get_train_config(): {**TRAINING_CONFIG, **AUGMENTATION_CONFIG, **kwargs}
# =============================================================================
AUGMENTATION_CONFIG = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.6,
    "degrees": 5.0,
    "translate": 0.2,
    "scale": 0.25,
    "shear": 2.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "auto_augment": "",
    "erasing": 0.0,
}

# Уніфікований пайплайн Albumentations; передається в model.train(augmentations=CUSTOM_TRANSFORMS)
# Albumentations 2.x: GaussNoise — std_range (не var_limit); RandomFog — fog_coef_range (не fog_coef_lower/upper)
CUSTOM_TRANSFORMS = [
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.GaussNoise(std_range=(0.05, 0.2), p=0.4),
    A.ISONoise(color_shift=(0.01, 0.01), intensity=(0.1, 0.5), p=0.3),
    A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.1, p=0.2),
    A.PixelDropout(dropout_prob=0.01, per_channel=False, p=0.2),
]


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


def _albu_transform_to_info(t) -> dict:
    """З об'єкта Albumentations трансформа витягує name, p та params для YAML."""
    out = {"name": type(t).__name__, "p": getattr(t, "p", None)}
    try:
        names = t.get_transform_init_args_names()
        out["params"] = {k: getattr(t, k, None) for k in names if k != "p"}
        # tuple -> list для YAML
        for k, v in out["params"].items():
            if isinstance(v, tuple):
                out["params"][k] = list(v)
    except Exception:
        out["params"] = {}
    return out


def get_albu_augmentations_info():
    """Опис CUSTOM_TRANSFORMS для збереження в augmentations_info.yaml (один джерело — константа CUSTOM_TRANSFORMS)."""
    if CUSTOM_TRANSFORMS is None:
        return None
    return {
        "description": "Custom Albumentations pipeline (list from global CUSTOM_TRANSFORMS)",
        "transforms": [_albu_transform_to_info(t) for t in CUSTOM_TRANSFORMS],
    }


def _write_augmentations_info_to_path(path: str, info: dict) -> None:
    """Записує словник info у path (YAML)."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(info, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _save_augmentations_info(save_dir: str, *, resume: bool = False) -> None:
    """Записує опис CUSTOM_TRANSFORMS у save_dir; ім'я файлу augmentations_info_resume.yaml при resume, інакше augmentations_info.yaml."""
    if CUSTOM_TRANSFORMS is None:
        return
    info = get_albu_augmentations_info()
    if info is None:
        return
    name = "augmentations_info_resume.yaml" if resume else "augmentations_info.yaml"
    path = os.path.join(save_dir, name)
    _write_augmentations_info_to_path(path, info)
    print(f"[Config] Опис Albumentations збережено: {path}")


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
    Збирає конфіг: TRAINING_CONFIG + AUGMENTATION_CONFIG + kwargs, потім фільтр по MODEL_TYPE.
    У train_model() ще додається config["augmentations"] = CUSTOM_TRANSFORMS.

    Args:
        **kwargs: Параметри, що перезаписують TRAINING_CONFIG /  AUGMENTATION_CONFIG

    Returns:
        dict: Базовий конфіг для model.train() (augmentations додається в train_model).
    """
    config = {**TRAINING_CONFIG, **AUGMENTATION_CONFIG, **kwargs}

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
    
    os.makedirs(RUNS_DIR, exist_ok=True)
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
    # Збір фінального конфігу: TRAINING_CONFIG + kwargs (фільтр по MODEL_TYPE),
    # потім дописуємо глобальний пайплайн Albumentations → один dict для model.train(**config)
    config = get_train_config(**kwargs)
    config["augmentations"] = CUSTOM_TRANSFORMS

    # Бекап опису аугментацій у project/ — буде доступний навіть при перериванні тренування
    info = get_albu_augmentations_info()
    if info is not None:
        latest_path = os.path.join(config["project"], "augmentations_info_latest.yaml")
        os.makedirs(config["project"], exist_ok=True)
        _write_augmentations_info_to_path(latest_path, info)

    print(f"Завантаження моделі: {model_path}")
    model = load_model(model_path)
    
    print("Початок навчання...")
    print(f"Архітектура: {MODEL_TYPE.upper()}")
    print(f"Модель: {model_path}")
    print(f"Конфігурація: epochs={config['epochs']}, batch={config['batch']}, imgsz={config['imgsz']}")
    print(f"Оптимізатор: {config['optimizer']}, lr0={config['lr0']}, cos_lr={config['cos_lr']}")
    
    # Ultralytics розв'язує path з data.yaml відносно cwd, а не відносно папки yaml — підставляємо абсолютний dataset root
    data_yaml_path = config["data"]
    try:
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Не вдалося прочитати data.yaml: {data_yaml_path}: {e}") from e
    data_cfg["path"] = os.path.abspath(DATASET_ROOT)
    tmp_fd, tmp_yaml = tempfile.mkstemp(suffix=".yaml", prefix="yolo_data_")
    try:
        os.close(tmp_fd)
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.dump(data_cfg, f, default_flow_style=False, allow_unicode=True)
        config["data"] = tmp_yaml
        # Навчання моделі (augmentations передаються для YOLO та RT-DETR; при несумісності — зрозуміла помилка)
        try:
            results = model.train(**config)
        except TypeError as e:
            if "augmentations" in str(e).lower() or "unexpected keyword" in str(e).lower():
                raise RuntimeError(
                    f"Помилка при виклику model.train(): ймовірно, параметр 'augmentations' "
                    f"(кастомні Albumentations) не підтримується для MODEL_TYPE='{MODEL_TYPE}'. "
                    f"Оригінальна помилка: {e}"
                ) from e
            raise
        except Exception as e:
            raise RuntimeError(
                f"Помилка навчання (MODEL_TYPE='{MODEL_TYPE}'): {e}. "
                f"Перевірте параметри конфігу, зокрема 'augmentations' при використанні RT-DETR."
            ) from e
    finally:
        if os.path.isfile(tmp_yaml):
            try:
                os.remove(tmp_yaml)
            except OSError:
                pass

    # Фактичну теку запуску створив Ultralytics (results.save_dir); туди пишемо augmentations_info[ _resume].yaml
    save_dir = str(results.save_dir)
    _save_augmentations_info(save_dir, resume=config.get("resume", False))
    # Після успішного завершення прибираємо бекап із project/ — він більше не потрібен
    latest_path = os.path.join(config["project"], "augmentations_info_latest.yaml")
    if os.path.isfile(latest_path):
        try:
            os.remove(latest_path)
        except OSError:
            pass

    trained_model_path = os.path.join(save_dir, "weights", "best.pt")

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
