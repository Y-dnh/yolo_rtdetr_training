"""
Модуль для тестування/валідації YOLO / RT-DETR моделі детекції на IR зображеннях.
Усі параметри конфігурації знаходяться на початку файлу.

Перемикач MODEL_TYPE дозволяє обрати архітектуру:
  - "yolo"   -> ultralytics.YOLO
  - "rtdetr"  -> ultralytics.RTDETR
"""

import os
import json
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO, RTDETR


# =============================================================================
# ВИБІР АРХІТЕКТУРИ: "yolo" або "rtdetr"
# =============================================================================
VALID_MODEL_TYPES = {"yolo", "rtdetr"}
MODEL_TYPE = "yolo"        # <-- ПЕРЕМИКАЧ: "yolo" або "rtdetr"

# =============================================================================
# ПАРАМЕТРИ, СПЕЦИФІЧНІ ДЛЯ КОЖНОЇ АРХІТЕКТУРИ (валідація)
# =============================================================================

# Ключі валідації, які є ТІЛЬКИ у YOLO
YOLO_ONLY_VAL_KEYS = {
    "agnostic_nms",     # Class-agnostic NMS — RT-DETR не використовує NMS
    "dnn",              # OpenCV DNN backend — тільки для YOLO
}

# Ключі валідації, які є ТІЛЬКИ у RT-DETR
RTDETR_ONLY_VAL_KEYS: set[str] = set()

# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ
# =============================================================================
PROJECT_NAME = "yolo26s_yaml"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, PROJECT_NAME)
# DATASET_ROOT = os.path.join(BASE_DIR, "dataset_split")
DATASET_ROOT = "D:/dataset_for_training"
YAML_PATH = os.path.join(DATASET_ROOT, "data.yaml")
EXPERIMENT_NAME = "validation_26s_yaml"

TRAINED_MODEL_PATH = os.path.join(PROJECT_DIR, "baseline2", "weights", "best.pt")

# Класи датасету
CLASSES = {
    0: "person",
    1: "car",
    2: "truck",
}


# =============================================================================
# ПАРАМЕТРИ ВАЛІДАЦІЇ (передаються як **kwargs до model.val())
# =============================================================================
VALIDATION_CONFIG = {
    # Параметри датасету
    "data": YAML_PATH,
    "split": "val",
    
    # Параметри детекції
    "conf": 0.25,
    "iou": 0.5,
    "imgsz": 1024,
    "device": None,  # Автоматичне визначення
    "batch": 8,  # Зменшено для економії пам'яті
    "max_det": 300,
    
    # Параметри обробки
    "rect": True,
    "half": True,
    "augment": False,
    "agnostic_nms": False,   # [YOLO-only] RT-DETR не використовує NMS
    "classes": None,
    "single_cls": False,
    "dnn": False,
    
    # Параметри виводу
    "save_json": True,
    "save_txt": False,
    "save_conf": True,
    "plots": True,
    "verbose": False,
    "workers": 8,  # 0 щоб уникнути multiprocessing та проблем з пам'яттю
    
    # Візуалізація
    "visualize": True,
    
    # Налаштування проекту
    "project": PROJECT_DIR,
    "name": EXPERIMENT_NAME,
}


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


def get_val_config(**kwargs) -> dict:
    """
    Повертає відфільтрований validation config для поточного MODEL_TYPE.
    
    Args:
        **kwargs: Параметри, що перезаписують VALIDATION_CONFIG
    
    Returns:
        dict: Готовий конфіг для model.val()
    """
    config = {**VALIDATION_CONFIG, **kwargs}

    if MODEL_TYPE == "rtdetr":
        return filter_config(config, YOLO_ONLY_VAL_KEYS)
    elif MODEL_TYPE == "yolo":
        return filter_config(config, RTDETR_ONLY_VAL_KEYS)
    return config


def print_header(model_path: str, config: dict, device: str) -> None:
    """Виведення заголовку валідації."""
    model_name = Path(model_path).name
    print()
    print("=" * 70)
    print("YOLO VALIDATION")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Dataset: {config['data']}")
    print(f"Split: {config['split']}")
    print(f"Image Size: {config['imgsz']}")
    print(f"Conf threshold: {config['conf']}")
    print(f"IoU threshold: {config['iou']}")
    print(f"Max detections: {config['max_det']}")
    print(f"Half (FP16): {config['half']}")
    print(f"Device: {device}")
    print("=" * 70)
    print()


def setup_device() -> str:
    """Визначення доступного пристрою."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def validate_model(
    model_path: str = TRAINED_MODEL_PATH,
    **kwargs
) -> object:
    """
    Валідація YOLO моделі на тестовому датасеті.
    
    Args:
        model_path: Шлях до навченої моделі
        **kwargs: Параметри валідації (перезаписують VALIDATION_CONFIG)
    
    Returns:
        object: Результати валідації
    """
    # Отримуємо відфільтрований конфіг для поточного MODEL_TYPE
    config = get_val_config(**kwargs)
    
    # Автоматичне визначення device якщо не вказано
    if config["device"] is None:
        config["device"] = setup_device()
    
    # Виводимо заголовок
    print_header(model_path, config, config["device"])
    
    print(f"[Validator] Loading {MODEL_TYPE.upper()} model from {Path(model_path).name}...")
    model = load_model(model_path)
    print(f"[Validator] Model loaded successfully!")
    
    # Запуск валідації
    results = model.val(**config)
    
    return results


def extract_metrics(validation_results: object) -> dict:
    """
    Витягування метрик з результатів валідації.
    
    Args:
        validation_results: Результати валідації від model.val()
    
    Returns:
        dict: Словник з метриками
    """
    precision = float(validation_results.box.mp) if hasattr(validation_results.box, "mp") else 0.0
    recall = float(validation_results.box.mr) if hasattr(validation_results.box, "mr") else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        # Основні метрики
        "mAP50": float(validation_results.box.map50) if hasattr(validation_results.box, "map50") else 0.0,
        "mAP50-95": float(validation_results.box.map) if hasattr(validation_results.box, "map") else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    # Per-class статистика
    class_stats = {}
    if hasattr(validation_results.box, "maps") and validation_results.box.maps is not None:
        maps = validation_results.box.maps
        for i, (class_id, class_name) in enumerate(CLASSES.items()):
            if i < len(maps):
                stat = {"mAP50": float(maps[i]) if maps[i] is not None else 0.0}
                if hasattr(validation_results.box, "p") and validation_results.box.p is not None:
                    if i < len(validation_results.box.p):
                        stat["precision"] = float(validation_results.box.p[i])
                if hasattr(validation_results.box, "r") and validation_results.box.r is not None:
                    if i < len(validation_results.box.r):
                        stat["recall"] = float(validation_results.box.r[i])
                class_stats[str(class_id)] = stat
    
    metrics["class_stats"] = class_stats
    
    return metrics


def get_speed_info(validation_results: object) -> dict:
    """Отримання інформації про швидкість."""
    speed_info = {}
    if hasattr(validation_results, "speed"):
        speed = validation_results.speed
        speed_info = {
            "preprocess_ms": speed.get("preprocess", 0),
            "inference_ms": speed.get("inference", 0),
            "postprocess_ms": speed.get("postprocess", 0),
        }
        total_time = sum(speed_info.values())
        speed_info["total_ms"] = total_time
        speed_info["fps"] = round(1000 / total_time, 2) if total_time > 0 else 0
    return speed_info


def save_results_json(
    metrics: dict,
    speed_info: dict,
    model_path: str,
    output_dir: str
) -> str:
    """
    Збереження результатів у єдиний JSON файл.
    
    Args:
        metrics: Метрики валідації
        speed_info: Інформація про швидкість
        model_path: Шлях до моделі
        output_dir: Директорія для збереження
    
    Returns:
        str: Шлях до збереженого файлу
    """
    results = {
        "metrics": {
            "mAP50": metrics["mAP50"],
            "mAP50-95": metrics["mAP50-95"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "class_stats": metrics.get("class_stats", {}),
        },
        "num_classes": len(CLASSES),
        "classes": list(CLASSES.values()),
        "inference_fps": speed_info.get("fps", 0),
        "inference_latency_ms": speed_info.get("total_ms", 0),
        "split": VALIDATION_CONFIG["split"],
        "dataset_dir": DATASET_ROOT,
        "validation_date": datetime.now().isoformat(),
        "inference_config": {
            "conf_threshold": VALIDATION_CONFIG["conf"],
            "iou_threshold": VALIDATION_CONFIG["iou"],
            "max_det": VALIDATION_CONFIG["max_det"],
            "classes": VALIDATION_CONFIG["classes"],
            "agnostic_nms": VALIDATION_CONFIG["agnostic_nms"],
            "half": VALIDATION_CONFIG["half"],
            "batch_size": VALIDATION_CONFIG["batch"],
            "imgsz": VALIDATION_CONFIG["imgsz"],
            "workers": VALIDATION_CONFIG["workers"],
            "device": VALIDATION_CONFIG["device"] or "cuda",
        },
        "model_path": model_path,
    }
    
    json_path = os.path.join(output_dir, "validation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return json_path


def generate_markdown_report(
    metrics: dict,
    speed_info: dict,
    model_path: str,
    output_dir: str
) -> str:
    """
    Генерація детального markdown звіту.
    
    Args:
        metrics: Словник з метриками
        speed_info: Інформація про швидкість
        model_path: Шлях до моделі
        output_dir: Директорія для збереження звіту
    
    Returns:
        str: Шлях до збереженого звіту
    """
    model_name = Path(model_path).stem
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Формуємо звіт у форматі як у прикладі
    report_content = f"""# 🎯 YOLO Validation Report

## Experiment Overview

| **Parameter** | **Value** |
|---------------|-----------|
| **Model** | `{model_name}` |
| **Model Path** | `{model_path}` |
| **Date & Time** | {current_time} |
| **Dataset** | `{DATASET_ROOT}` |
| **Split** | `{VALIDATION_CONFIG['split']}` |
| **Object Categories** | {len(CLASSES)} |

## Configuration Settings

| **Setting** | **Value** |
|-------------|-----------|
| **Confidence Threshold** | {VALIDATION_CONFIG['conf']} |
| **IoU Threshold** | {VALIDATION_CONFIG['iou']} |
| **Image Size** | {VALIDATION_CONFIG['imgsz']} |
| **Batch Size** | {VALIDATION_CONFIG['batch']} |
| **Half (FP16)** | {VALIDATION_CONFIG['half']} |
| **Device** | {VALIDATION_CONFIG['device'] or 'cuda'} |

---

## 📊 Overall Performance

| **Metric** | **Value** |
|------------|-----------|
| **mAP@0.5** | {metrics['mAP50']:.4f} |
| **mAP@0.5:0.95** | {metrics['mAP50-95']:.4f} |
| **Precision** | {metrics['precision']:.4f} |
| **Recall** | {metrics['recall']:.4f} |
| **F1 Score** | {metrics['f1']:.4f} |

---

## 📋 Per-Class Performance

| **Class** | **mAP@0.5** | **Precision** | **Recall** |
|-----------|-------------|---------------|------------|
"""
    
    # Додаємо per-class метрики
    class_stats = metrics.get("class_stats", {})
    for class_id, class_name in CLASSES.items():
        stat = class_stats.get(str(class_id), {})
        mAP = stat.get("mAP50", 0)
        prec = stat.get("precision", 0)
        rec = stat.get("recall", 0)
        report_content += f"| {class_name} | {mAP:.4f} | {prec:.4f} | {rec:.4f} |\n"
    
    report_content += f"""
---

## ⚡ Inference Speed

| **Metric** | **Value** |
|------------|-----------|
| **FPS** | {speed_info.get('fps', 0):.1f} |
| **Latency** | {speed_info.get('total_ms', 0):.2f} ms/image |
| **Preprocess** | {speed_info.get('preprocess_ms', 0):.2f} ms |
| **Inference** | {speed_info.get('inference_ms', 0):.2f} ms |
| **Postprocess** | {speed_info.get('postprocess_ms', 0):.2f} ms |

---

*📊 Report generated by YOLO Validation System*  
*🕐 {current_time}*
"""
    
    # Зберігаємо звіт
    report_path = os.path.join(output_dir, "validation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    return report_path


def print_summary(metrics: dict, speed_info: dict, json_path: str, report_path: str) -> None:
    """Виведення підсумку у консоль."""
    print()
    print(f"📊 Inference Speed: {speed_info.get('fps', 0):.1f} FPS ({speed_info.get('total_ms', 0):.2f} ms/image)")
    print()
    print("=" * 50)
    print("YOLO VALIDATION SUMMARY")
    print("=" * 50)
    print(f"mAP@0.5:      {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1 Score:     {metrics['f1']:.4f}")
    print(f"JSON Results: {json_path}")
    print(f"MD Report:    {report_path}")
    print("=" * 50)
    print()
    print(f"[Results] Результати збережено: {json_path}")


def save_results(
    validation_results: object,
    metrics: dict,
    model_path: str = TRAINED_MODEL_PATH,
    output_dir: str = None
) -> dict:
    """
    Збереження результатів валідації.
    
    Args:
        validation_results: Результати валідації
        metrics: Витягнуті метрики
        model_path: Шлях до моделі (для звіту)
        output_dir: Директорія для збереження
    
    Returns:
        dict: Словник зі шляхами до збережених файлів
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_DIR, EXPERIMENT_NAME)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Отримуємо інформацію про швидкість
    speed_info = get_speed_info(validation_results)
    
    saved_files = {}
    
    # Збереження результатів у єдиний JSON файл
    json_path = save_results_json(metrics, speed_info, model_path, output_dir)
    saved_files["validation_json"] = json_path
    
    # Генерація markdown звіту
    report_path = generate_markdown_report(metrics, speed_info, model_path, output_dir)
    saved_files["markdown_report"] = report_path
    
    # Виведення підсумку
    print_summary(metrics, speed_info, json_path, report_path)
    
    return saved_files


def main(
    model_path: str = TRAINED_MODEL_PATH,
    save_results_flag: bool = True,
    **kwargs
):
    """
    Головна функція для запуску валідації.
    
    Args:
        model_path: Шлях до навченої моделі
        save_results_flag: Чи зберігати результати у файли
        **kwargs: Додаткові параметри валідації
    """
    # Запуск валідації
    results = validate_model(model_path=model_path, **kwargs)
    
    # Витягування метрик
    metrics = extract_metrics(results)
    
    # Збереження результатів
    if save_results_flag:
        save_results(results, metrics, model_path=model_path)
    
    return results, metrics


if __name__ == "__main__":
    main()
