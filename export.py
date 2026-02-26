"""
Модуль для експорту YOLO / RT-DETR моделей у різні формати.
Підтримує: ONNX, TensorRT, TFLite, OpenVINO, CoreML та інші.
Усі параметри конфігурації знаходяться на початку файлу.
Документація: https://docs.ultralytics.com/modes/export/#arguments

Перемикач MODEL_TYPE дозволяє обрати архітектуру:
  - "yolo"   -> ultralytics.YOLO
  - "rtdetr"  -> ultralytics.RTDETR
"""

import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO, RTDETR


# =============================================================================
# ВИБІР АРХІТЕКТУРИ: "yolo" або "rtdetr"
# =============================================================================
VALID_MODEL_TYPES = {"yolo", "rtdetr"}
MODEL_TYPE = "yolo"        # <-- ПЕРЕМИКАЧ: "yolo" або "rtdetr"

# =============================================================================
# ПАРАМЕТРИ, СПЕЦИФІЧНІ ДЛЯ КОЖНОЇ АРХІТЕКТУРИ (експорт)
# =============================================================================

# Ключі експорту, які є ТІЛЬКИ у YOLO
YOLO_ONLY_EXPORT_KEYS = {
    "nms",              # RT-DETR — NMS-free архітектура
}

# Ключі експорту, які є ТІЛЬКИ у RT-DETR
RTDETR_ONLY_EXPORT_KEYS: set[str] = set()

# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ
# =============================================================================
PROJECT_NAME = "yolo26s"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, PROJECT_NAME)
DATASET_ROOT = os.path.join(BASE_DIR, "dataset_split")
YAML_PATH = os.path.join(DATASET_ROOT, "yolo.yaml")

# Шлях до моделі для експорту
MODEL_PATH = os.path.join(PROJECT_DIR, "baseline", "weights", "best.pt")


# =============================================================================
# КОНФІГУРАЦІЯ ЕКСПОРТУ
# Документація: https://docs.ultralytics.com/modes/export/#arguments
# =============================================================================
EXPORT_CONFIG = {
    "format": "engine",
    
    "imgsz": (1024, 1024),
    "half": True,
    "int8": False,
    "optimize": False,
    "dynamic": False,
    "simplify": True,
    "opset": 12,
    "workspace": None,
    "nms": False,               # [YOLO-only] Вбудувати NMS (RT-DETR: NMS-free архітектура)
    "batch": 1,
    "device": None,
    "data": None,
    "fraction": 1.0,
    "keras": False,
    "end2end": None,
}


# =============================================================================
# ФУНКЦІЇ
# =============================================================================
def validate_model_type() -> None:
    """Перевірка що MODEL_TYPE має допустиме значення."""
    if MODEL_TYPE not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Невідомий MODEL_TYPE: '{MODEL_TYPE}'. "
            f"Допустимі значення: {sorted(VALID_MODEL_TYPES)}"
        )


def load_model(model_path: str):
    """
    Завантаження моделі відповідно до MODEL_TYPE.
    Автоматично визначає тип, якщо в шляху є 'rtdetr'.
    
    Args:
        model_path: Шлях до моделі
    
    Returns:
        Завантажена модель (YOLO або RTDETR)
    
    Raises:
        ValueError: Якщо MODEL_TYPE невідомий
    """
    validate_model_type()

    if MODEL_TYPE == "rtdetr" or "rtdetr" in model_path.lower():
        print(f"[Model] Завантаження RT-DETR: {model_path}")
        return RTDETR(model_path)
    else:
        print(f"[Model] Завантаження YOLO: {model_path}")
        return YOLO(model_path)


def filter_config(config: dict, excluded_keys: set) -> dict:
    """
    Фільтрує конфігурацію: видаляє ключі, несумісні з поточною архітектурою.
    
    Args:
        config: Вхідний словник конфігурації
        excluded_keys: Множина ключів, які потрібно видалити
    
    Returns:
        dict: Відфільтрований словник
    """
    removed = set(config.keys()) & excluded_keys
    if removed:
        print(f"[Config] MODEL_TYPE='{MODEL_TYPE}' -> видалено несумісні ключі: {sorted(removed)}")

    return {k: v for k, v in config.items() if k not in excluded_keys}


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
        return filter_config(config, RTDETR_ONLY_EXPORT_KEYS)
    return config


def print_header(model_path: str, config: dict) -> None:
    """Виведення заголовку експорту."""
    model_name = Path(model_path).name
    print()
    print("=" * 70)
    print("YOLO MODEL EXPORT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Format: {config['format']}")
    print(f"Image Size: {config['imgsz']}")
    print(f"Dynamic: {config.get('dynamic', False)}")
    print(f"Half (FP16): {config.get('half', False)}")
    print(f"INT8: {config.get('int8', False)}")
    if config['format'] == 'onnx':
        print(f"Simplify: {config.get('simplify', True)}")
        print(f"Opset: {config.get('opset', 'auto')}")
    if config['format'] == 'engine':
        print(f"Workspace: {config.get('workspace', 'auto')} GiB")
    if config.get('int8') and config.get('data'):
        print(f"Calibration data: {config['data']}")
    print("=" * 70)
    print()


def export_model(
    model_path: str = MODEL_PATH,
    **kwargs
) -> str | None:
    """
    Експорт YOLO моделі.
    
    Args:
        model_path: Шлях до моделі (.pt файл)
        **kwargs: Параметри експорту (перезаписують EXPORT_CONFIG)
    
    Returns:
        str | None: Шлях до експортованої моделі або None при помилці
    """
    # Перевірка існування моделі
    if not os.path.exists(model_path):
        print(f"Помилка: Модель не знайдена: {model_path}")
        return None
    
    # Отримуємо відфільтрований конфіг для поточного MODEL_TYPE
    config = get_export_config(**kwargs)
    
    # Фільтруємо None значення (ultralytics не любить None)
    export_params = {k: v for k, v in config.items() if v is not None}
    
    # Виводимо заголовок
    print_header(model_path, config)
    
    # Завантаження та експорт
    start_time = datetime.now()
    print(f"[Export] Завантаження моделі: {Path(model_path).name}")
    
    try:
        model = load_model(model_path)
        print(f"[Export] Експорт у формат '{config['format']}'...")
        
        exported_path = model.export(**export_params)
        
        export_time = (datetime.now() - start_time).total_seconds()
        
        # Розмір файлу
        if exported_path and os.path.exists(exported_path):
            if os.path.isfile(exported_path):
                size_mb = os.path.getsize(exported_path) / (1024 * 1024)
            else:
                # Для директорій
                total_size = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, files in os.walk(exported_path)
                    for f in files
                )
                size_mb = total_size / (1024 * 1024)
        else:
            size_mb = 0
        
        print()
        print("=" * 70)
        print("EXPORT COMPLETED")
        print("=" * 70)
        print(f"Output: {exported_path}")
        print(f"Size: {size_mb:.2f} MB")
        print(f"Time: {export_time:.2f} seconds")
        print("=" * 70)
        
        return exported_path
        
    except Exception as e:
        print(f"\nПомилка під час експорту: {e}")
        return None


def main():
    """Головна функція для запуску експорту."""
    result = export_model()
    return result


if __name__ == "__main__":
    main()
