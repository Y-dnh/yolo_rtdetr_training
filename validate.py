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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


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

TRAINED_MODEL_PATH = os.path.join(PROJECT_DIR, "baseline", "weights", "best.pt")

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
    "split": "test",
    
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


def _safe_float(x, default=0.0):
    """Повертає float або default якщо значення недоступне."""
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# Пороги COCO/YOLO для розміру об'єктів (площа в пікселях²)
COCO_AREA_SMALL = 32 ** 2   # area < 32²
COCO_AREA_MEDIUM_MAX = 96 ** 2  # 32²–96² = medium, > 96² = large


# GT у YOLO — у .txt; pycocotools потребує GT у COCO JSON → конвертуємо val у один файл.
def _image_size_fast(img_path: str) -> tuple[int, int] | None:
    """Повертає (width, height) зображення без повного завантаження."""
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.size
    except Exception:
        return None


def yolo_split_to_coco_annotations(
    data_yaml_path: str,
    output_json_path: str,
    split: str = "val",
    imgsz: int = None,
) -> str | None:
    """
    Конвертує val-спліт у форматі YOLO (лейбли .txt) у один COCO JSON.
    width/height і bbox — у координатах оригінального зображення (як у predictions при rect=True).
    """
    imgsz = imgsz or VALIDATION_CONFIG.get("imgsz", 640)
    img_paths, dataset_path = _load_yolo_val_data(data_yaml_path, split)
    if not img_paths or not dataset_path:
        return None
    images = []
    annotations = []
    ann_id = 0
    for img_id, img_path in enumerate(img_paths):
        if not os.path.isfile(img_path):
            continue
        wh = _image_size_fast(img_path)
        if wh is None:
            wh = (imgsz, imgsz)
        w_img, h_img = wh
        images.append({
            "id": img_id,
            "file_name": os.path.basename(img_path),
            "width": w_img,
            "height": h_img,
        })
        label_path = _yolo_label_path(img_path, dataset_path)
        if not os.path.isfile(label_path):
            continue
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                xc, yc, w_n, h_n = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except (ValueError, IndexError):
                continue
            x = (xc - w_n / 2) * w_img
            y = (yc - h_n / 2) * h_img
            w = w_n * w_img
            h = h_n * h_img
            area = w * h
            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls + 1,
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(area, 2),
                "iscrowd": 0,
            })
    categories = [{"id": i + 1, "name": name} for i, name in CLASSES.items()]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    try:
        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        return output_json_path
    except Exception:
        return None


def _ensure_predictions_coco_json(results, save_dir: str, img_paths: list, imgsz: int) -> str | None:
    """У ultralytics base validator при save_json і непорожньому jdict сам пише save_dir/predictions.json. Тут лише добираємо: якщо файлу немає (напр. jdict порожній) — пишемо з jdict або збираємо з model.predict."""
    pred_path = os.path.join(save_dir, "predictions.json")
    if os.path.isfile(pred_path):
        return pred_path
    jdict = getattr(results, "jdict", None)
    if jdict and len(jdict) > 0:
        try:
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(jdict, f, indent=2, ensure_ascii=False)
            return pred_path
        except Exception:
            pass
    model = getattr(results, "model", None)
    if not model or not img_paths:
        return None
    try:
        pred_results = model.predict(
            img_paths,
            imgsz=imgsz or VALIDATION_CONFIG.get("imgsz", 640),
            conf=VALIDATION_CONFIG.get("conf", 0.25),
            iou=VALIDATION_CONFIG.get("iou", 0.5),
            verbose=False,
            max_det=300,
        )
    except Exception:
        return None
    coco_preds = []
    for img_id, (img_path, result) in enumerate(zip(img_paths, pred_results)):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            w = x2 - x1
            h = y2 - y1
            coco_preds.append({
                "image_id": img_id,
                "category_id": int(clss[i]) + 1,
                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                "score": round(float(confs[i]), 4),
            })
    try:
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(coco_preds, f, indent=2, ensure_ascii=False)
        return pred_path
    except Exception:
        return None


def _load_yolo_val_data(data_yaml_path: str, split: str = "val"):
    """Повертає (list of image_paths, dataset_path) з data.yaml для YOLO. Якщо помилка — ([], None)."""
    try:
        import yaml
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return [], None
    path = data.get("path") or os.path.dirname(os.path.abspath(data_yaml_path))
    path = os.path.normpath(os.path.abspath(path))
    val_key = split if split in data else "val"
    val = data.get(val_key, "")
    if not val:
        return [], path
    val = os.path.normpath(val)
    if not os.path.isabs(val):
        val = os.path.join(path, val)
    if os.path.isfile(val) and val.endswith(".txt"):
        with open(val, "r", encoding="utf-8") as f:
            lines = [x.strip() for x in f if x.strip()]
        img_paths = [os.path.join(path, p) if not os.path.isabs(p) else p for p in lines]
    elif os.path.isdir(val):
        import glob as _g
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            img_paths.extend(_g.glob(os.path.join(val, ext)))
        img_paths.sort()
    else:
        img_paths = []
    return img_paths, path


def _yolo_label_path(img_path: str, dataset_path: str) -> str:
    """Шлях до .txt лейбла YOLO для зображення (images/ -> labels/, інший розширення .txt)."""
    rel = os.path.relpath(img_path, dataset_path)
    rel = rel.replace("\\", "/")
    for old in ("images/", "Images/", "image/"):
        if old in rel:
            rel = rel.replace(old, "labels/", 1)
            break
    else:
        rel = "labels/" + os.path.basename(rel)
    base, _ = os.path.splitext(rel)
    return os.path.join(dataset_path, base + ".txt")


def _find_coco_annotations(data_yaml_path: str, split: str = "val") -> str | None:
    """Шукає шлях до COCO annotations (instances_*.json) з data.yaml."""
    try:
        import yaml
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None
    path = data.get("path") or os.path.dirname(os.path.abspath(data_yaml_path))
    path = os.path.normpath(path)
    ann_dir = os.path.join(path, "annotations")
    if not os.path.isdir(ann_dir):
        return None
    for name in ("instances_val2017.json", "instances_val.json", f"instances_{split}.json"):
        p = os.path.join(ann_dir, name)
        if os.path.isfile(p):
            return p
    import glob as _glob
    candidates = _glob.glob(os.path.join(ann_dir, "instances_*.json"))
    return candidates[0] if candidates else None


def _extract_per_class_ap_by_area(coco_eval) -> list[dict]:
    """
    Витягує для кожного класу AP small / medium / large з coco_eval.eval['precision'].
    COCO precision shape: (T, R, K, A, M) — T=IoU, R=recall, K=classes, A=area (0:all,1:small,2:medium,3:large), M=maxDets.
    """
    import numpy as np
    out = []
    try:
        prec = coco_eval.eval.get("precision")
        if prec is None:
            return out
        prec = np.asarray(prec)
        # A: 0=all, 1=small, 2=medium, 3=large
        K = prec.shape[2]
        for k in range(K):
            ap_s = float(np.mean(prec[:, :, k, 1, :])) if prec.shape[3] > 1 else 0.0
            ap_m = float(np.mean(prec[:, :, k, 2, :])) if prec.shape[3] > 2 else 0.0
            ap_l = float(np.mean(prec[:, :, k, 3, :])) if prec.shape[3] > 3 else 0.0
            out.append({"small": ap_s, "medium": ap_m, "large": ap_l})
    except Exception:
        pass
    return out


def run_coco_eval_metrics(
    save_dir: str,
    data_yaml_path: str = None,
    split: str = None,
    annotation_path: str = None,
) -> dict:
    """
    Запускає COCO evaluation за predictions.json та annotations, повертає метрики.
    annotation_path: якщо задано — використовується цей JSON (наприклад з YOLO-конвертації);
    інакше шукаються COCO annotations через _find_coco_annotations(data_yaml_path).
    """
    split = split or VALIDATION_CONFIG.get("split", "val")
    empty = {
        "ap_by_size": {"small": 0.0, "medium": 0.0, "large": 0.0},
        "ar_maxdets1": 0.0,
        "ar_maxdets10": 0.0,
        "ar_maxdets100": 0.0,
        "ar_small": 0.0,
        "ar_medium": 0.0,
        "ar_large": 0.0,
        "ap_by_class_area": [],
    }
    pred_json = os.path.join(save_dir, "predictions.json")
    if not os.path.isfile(pred_json):
        print(f"[COCO eval] Немає файлу: {pred_json}", flush=True)
        return empty
    anno_json = annotation_path if (annotation_path and os.path.isfile(annotation_path)) else None
    if not anno_json and data_yaml_path:
        anno_json = _find_coco_annotations(data_yaml_path, split)
    if not anno_json or not os.path.isfile(anno_json):
        print(f"[COCO eval] Немає annotations: {anno_json or annotation_path}", flush=True)
        return empty
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        print(f"[COCO eval] Помилка: {e}", flush=True)
        print("[COCO eval] Встанови: pip install pycocotools  (AP by size, AR будуть 0 поки не встановлено)", flush=True)
        return empty
    try:
        anno = COCO(anno_json)
        # У предиктів ultralytics image_id = stem файлу; у наших annotations — 0,1,2,…. Зводимо до спільного id по file_name.
        with open(pred_json, "r", encoding="utf-8") as f:
            pred_list = json.load(f)
        images_list = anno.dataset.get("images", [])
        fn2id = {img["file_name"]: img["id"] for img in images_list}
        mapped = []
        for p in pred_list:
            fn = p.get("file_name")
            if fn and fn in fn2id:
                p = dict(p)
                p["image_id"] = fn2id[fn]
                mapped.append(p)
        if not mapped:
            print(f"[COCO eval] Пропущено: у annotations {len(fn2id)} images, у predictions {len(pred_list)} записів, збігів по file_name: 0. Перевір split.", flush=True)
            return empty
        pred = anno.loadRes(mapped)
        coco_eval = COCOeval(anno, pred, "bbox")
        img_ids = sorted(anno.getImgIds())
        if img_ids:
            coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # stats: 0=AP .5:.95, 1=AP .5, 2=AP .75, 3=AP small, 4=AP medium, 5=AP large,
        #        6=AR maxDets=1, 7=AR maxDets=10, 8=AR maxDets=100, 9=AR small, 10=AR medium, 11=AR large
        s = coco_eval.stats
        ap_by_class_area = _extract_per_class_ap_by_area(coco_eval)
        print(f"[COCO eval] OK: {len(fn2id)} images, {len(mapped)} predictions. AP small={s[3]:.4f}, medium={s[4]:.4f}, large={s[5]:.4f}", flush=True)
        return {
            "ap_by_size": {
                "small": float(s[3]) if len(s) > 3 else 0.0,
                "medium": float(s[4]) if len(s) > 4 else 0.0,
                "large": float(s[5]) if len(s) > 5 else 0.0,
            },
            "ar_maxdets1": float(s[6]) if len(s) > 6 else 0.0,
            "ar_maxdets10": float(s[7]) if len(s) > 7 else 0.0,
            "ar_maxdets100": float(s[8]) if len(s) > 8 else 0.0,
            "ar_small": float(s[9]) if len(s) > 9 else 0.0,
            "ar_medium": float(s[10]) if len(s) > 10 else 0.0,
            "ar_large": float(s[11]) if len(s) > 11 else 0.0,
            "ap_by_class_area": ap_by_class_area,
        }
    except Exception as e:
        print(f"[COCO eval] Помилка: {e}", flush=True)
        return empty


def _get_ap_by_size(validation_results: object) -> dict:
    """
    Спроба отримати AP small / medium / large з результатів валідації.
    Доступно лише при валідації COCO/LVIS датасету (save_json=True, is_coco).
    """
    out = {"small": None, "medium": None, "large": None}
    try:
        # Ultralytics зберігає ці метрики в stats після faster-coco-eval (COCO/LVIS)
        stats = getattr(validation_results, "get_stats", None)
        if callable(stats):
            s = stats()
            if isinstance(s, dict):
                out["small"] = s.get("metrics/mAP_small(B)")
                out["medium"] = s.get("metrics/mAP_medium(B)")
                out["large"] = s.get("metrics/mAP_large(B)")
        for k in list(out.keys()):
            if out[k] is not None:
                out[k] = _safe_float(out[k], None)
    except Exception:
        pass
    return out


def extract_metrics(validation_results: object) -> dict:
    """
    Витягування метрик з результатів валідації.
    
    Args:
        validation_results: Результати валідації від model.val()
    
    Returns:
        dict: Словник з метриками
    """
    box = validation_results.box
    precision = _safe_float(getattr(box, "mp", None))
    recall = _safe_float(getattr(box, "mr", None))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "mAP50": _safe_float(getattr(box, "map50", None)),
        "mAP50-95": _safe_float(getattr(box, "map", None)),
        "mAP75": _safe_float(getattr(box, "map75", None)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap_by_size": _get_ap_by_size(validation_results),
        "ar_maxdets1": 0.0,
        "ar_maxdets10": 0.0,
        "ar_maxdets100": 0.0,
        "ar_small": 0.0,
        "ar_medium": 0.0,
        "ar_large": 0.0,
        "ap_by_class_area": [],
    }

    # Per-class: mAP50, mAP50-95, precision, recall, F1 (індекс по ap_class_index)
    class_stats = {}
    ap_class_index = getattr(box, "ap_class_index", None)
    if hasattr(ap_class_index, "tolist"):
        ap_class_index = ap_class_index.tolist()
    elif not isinstance(ap_class_index, list):
        ap_class_index = list(range(len(CLASSES))) if ap_class_index is None else []

    maps = getattr(box, "maps", None)
    ap5095 = getattr(box, "ap", None)
    p_list = getattr(box, "p", None)
    r_list = getattr(box, "r", None)
    f1_list = getattr(box, "f1", None)

    for class_id, _ in CLASSES.items():
        try:
            idx = ap_class_index.index(int(class_id)) if ap_class_index else class_id
        except (ValueError, AttributeError):
            idx = int(class_id)
        m50 = None
        if maps is not None:
            if class_id < len(maps):
                m50 = maps[class_id]
            elif idx < len(maps):
                m50 = maps[idx]
        stat = {
            "mAP50": _safe_float(m50),
            "mAP50-95": _safe_float(ap5095[idx] if ap5095 is not None and idx < len(ap5095) else None),
            "precision": _safe_float(p_list[idx] if p_list is not None and idx < len(p_list) else None),
            "recall": _safe_float(r_list[idx] if r_list is not None and idx < len(r_list) else None),
            "f1": _safe_float(f1_list[idx] if f1_list is not None and idx < len(f1_list) else None),
        }
        if stat["f1"] == 0.0 and stat["precision"] and stat["recall"]:
            p, r = stat["precision"], stat["recall"]
            stat["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
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
    ap_size = metrics.get("ap_by_size") or {}
    results = {
        "metrics": {
            "mAP50": metrics["mAP50"],
            "mAP50-95": metrics["mAP50-95"],
            "mAP75": metrics.get("mAP75"),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "ap_small": ap_size.get("small"),
            "ap_medium": ap_size.get("medium"),
            "ap_large": ap_size.get("large"),
            "ar_maxdets1": metrics.get("ar_maxdets1"),
            "ar_maxdets10": metrics.get("ar_maxdets10"),
            "ar_maxdets100": metrics.get("ar_maxdets100"),
            "ar_small": metrics.get("ar_small"),
            "ar_medium": metrics.get("ar_medium"),
            "ar_large": metrics.get("ar_large"),
            "class_stats": metrics.get("class_stats", {}),
            "ap_by_class_area": metrics.get("ap_by_class_area", []),
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


def _plot_per_class_metrics(metrics: dict, output_dir: str) -> str | None:
    """
    Будує bar chart mAP50 та mAP50-95 по класах, зберігає в output_dir.
    Повертає ім'я файлу або None якщо побудова неможлива.
    """
    if not _HAS_MATPLOTLIB:
        return None
    class_stats = metrics.get("class_stats", {})
    if not class_stats:
        return None
    names = [CLASSES[i] for i in range(len(CLASSES))]
    m50 = [class_stats.get(str(i), {}).get("mAP50", 0) for i in range(len(CLASSES))]
    m5095 = [class_stats.get(str(i), {}).get("mAP50-95", 0) for i in range(len(CLASSES))]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, m50, w, label="mAP@0.5", color="steelblue")
    ax.bar(x + w / 2, m5095, w, label="mAP@0.5:0.95", color="coral", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("Per-class mAP")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(output_dir, "per_class_map.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return "per_class_map.png"


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
    per_class_plot = _plot_per_class_metrics(metrics, output_dir)

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
| **mAP@0.75** | {metrics.get('mAP75', 0):.4f} |
| **Precision** | {metrics['precision']:.4f} |
| **Recall** | {metrics['recall']:.4f} |
| **F1 Score** | {metrics['f1']:.4f} |

---
"""
    # AP за розміром об'єктів (small / medium / large) — загалом
    ap_size = metrics.get("ap_by_size") or {}
    ap_s, ap_m, ap_l = ap_size.get("small"), ap_size.get("medium"), ap_size.get("large")
    def _fmt_ap(v):
        return f"{v:.4f}" if v is not None else "—"
    report_content += """
## 📐 AP за розміром об'єктів (overall)

| **Розмір** | **AP@0.5:0.95** | **Примітка** |
|------------|-----------------|--------------|
| **Small** (area < 32² px) | """ + _fmt_ap(ap_s) + """ | Маленькі об'єкти |
| **Medium** (32²–96² px) | """ + _fmt_ap(ap_m) + """ | Середні об'єкти |
| **Large** (area ≥ 96² px) | """ + _fmt_ap(ap_l) + """ | Великі об'єкти |

"""
    if ap_s is None and ap_m is None and ap_l is None:
        report_content += "*AP за розміром доступні при валідації датасету у форматі COCO (save_json=True, annotations у COCO).*\n\n---\n\n"
    else:
        report_content += "---\n\n"

    # AR (Average Recall) — додаткові метрики COCO
    ar1 = metrics.get("ar_maxdets1", 0)
    ar10 = metrics.get("ar_maxdets10", 0)
    ar100 = metrics.get("ar_maxdets100", 0)
    ar_s = metrics.get("ar_small", 0)
    ar_m = metrics.get("ar_medium", 0)
    ar_l = metrics.get("ar_large", 0)
    report_content += """
### Додаткові метрики (COCO): AR — Average Recall

| **Метрика** | **Значення** |
|-------------|--------------|
| **AR @ maxDets=1** | """ + f"{ar1:.4f}" + """ |
| **AR @ maxDets=10** | """ + f"{ar10:.4f}" + """ |
| **AR @ maxDets=100** | """ + f"{ar100:.4f}" + """ |
| **AR small** | """ + f"{ar_s:.4f}" + """ |
| **AR medium** | """ + f"{ar_m:.4f}" + """ |
| **AR large** | """ + f"{ar_l:.4f}" + """ |

---

"""
    # AP по класах за розміром об'єкта
    ap_by_class_area = metrics.get("ap_by_class_area") or []
    if ap_by_class_area:
        report_content += """## AP по класах за розміром об'єкта

| **Class** | **AP small** | **AP medium** | **AP large** |
|-----------|---------------|---------------|--------------|
"""
        for i, class_name in CLASSES.items():
            row = ap_by_class_area[i] if i < len(ap_by_class_area) else {}
            report_content += "| {} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                class_name,
                row.get("small", 0),
                row.get("medium", 0),
                row.get("large", 0),
            )
        report_content += "\n---\n\n"

    report_content += """## 📋 Per-Class Performance

| **Class** | **mAP@0.5** | **mAP@0.5:0.95** | **Precision** | **Recall** | **F1** |
|-----------|-------------|------------------|---------------|-----------|--------|
"""
    class_stats = metrics.get("class_stats", {})
    for class_id, class_name in CLASSES.items():
        stat = class_stats.get(str(class_id), {})
        m50 = stat.get("mAP50", 0)
        m5095 = stat.get("mAP50-95", 0)
        prec = stat.get("precision", 0)
        rec = stat.get("recall", 0)
        f1 = stat.get("f1", 0)
        report_content += f"| {class_name} | {m50:.4f} | {m5095:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} |\n"

    report_content += """
### Інтерпретація per-class метрик

- **mAP@0.5** — середня точність при IoU 0.5 (м’якший критерій).
- **mAP@0.5:0.95** — середня точність по IoU 0.5–0.95 (жорсткіший, COCO-style).
- **Precision** — частка коректних детекцій серед усіх передбачень класу.
- **Recall** — частка знайдених об’єктів класу серед усіх GT.
- **F1** — баланс між precision та recall.

---

## ⚡ Inference Speed

| **Metric** | **Value** |
|------------|-----------|
| **FPS** | """ + f"{speed_info.get('fps', 0):.1f}" + """ |
| **Latency** | """ + f"{speed_info.get('total_ms', 0):.2f}" + """ ms/image |
| **Preprocess** | """ + f"{speed_info.get('preprocess_ms', 0):.2f}" + """ ms |
| **Inference** | """ + f"{speed_info.get('inference_ms', 0):.2f}" + """ ms |
| **Postprocess** | """ + f"{speed_info.get('postprocess_ms', 0):.2f}" + """ ms |

---

## 📈 Графіки та додаткові метрики

"""
    if per_class_plot:
        report_content += f"**Per-class mAP:**\n\n![Per-class mAP]({per_class_plot})\n\n"
    # Посилання на графіки, згенеровані Ultralytics (plots=True); detect task часто з префіксом Box
    plot_names = [
        ("confusion_matrix.png", "Confusion Matrix"),
        ("BoxPR_curve.png", "Precision-Recall curve"),
        ("BoxF1_curve.png", "F1 Score vs Confidence"),
        ("BoxP_curve.png", "Precision vs Confidence"),
        ("BoxR_curve.png", "Recall vs Confidence"),
        ("PR_curve.png", "PR curve (alt)"),
        ("F1_curve.png", "F1 curve (alt)"),
        ("confusion_matrix_normalized.png", "Confusion Matrix (normalized)"),
    ]
    report_content += "### Графіки та візуалізації\n\nГрафіки з валідації (якщо збережені в цій директорії):\n\n"
    for fname, desc in plot_names:
        report_content += f"- **{desc}**: [{fname}]({fname})\n"
    report_content += """

### Корисні метрики для аналізу

| Метрика | Навіщо |
|---------|--------|
| **mAP@0.5** | Загальна якість при типовому IoU. |
| **mAP@0.75** | Строгіша локалізація боксів. |
| **mAP@0.5:0.95** | Зведена оцінка (COCO standard). |
| **Precision** | Важлива, якщо критичні false positives. |
| **Recall** | Важлива, якщо потрібно не пропускати об'єкти. |
| **F1** | Баланс між precision і recall. |
| **AP small/medium/large** | Якість на різних масштабах об'єктів (COCO). |

---

*📊 Report generated by YOLO Validation System*  
*🕐 """ + current_time + """*
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
    if metrics.get("mAP75") is not None:
        print(f"mAP@0.75:     {metrics['mAP75']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1 Score:     {metrics['f1']:.4f}")
    ap_size = metrics.get("ap_by_size") or {}
    print("AP by size:   small={}, medium={}, large={}".format(
        f"{ap_size.get('small') or 0:.4f}" if ap_size.get("small") is not None else "—",
        f"{ap_size.get('medium') or 0:.4f}" if ap_size.get("medium") is not None else "—",
        f"{ap_size.get('large') or 0:.4f}" if ap_size.get("large") is not None else "—",
    ))
    print("AR:           maxDets1={:.4f}, maxDets10={:.4f}, maxDets100={:.4f}".format(
        metrics.get("ar_maxdets1") or 0, metrics.get("ar_maxdets10") or 0, metrics.get("ar_maxdets100") or 0
    ))
    print("AR by size:   small={:.4f}, medium={:.4f}, large={:.4f}".format(
        metrics.get("ar_small") or 0, metrics.get("ar_medium") or 0, metrics.get("ar_large") or 0
    ))
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
    # Не створюємо папку і не передаємо save_dir: ultralytics сам створить одну папку за запуск (val, val2, val3…).
    # Раніше ми робили makedirs до val і передавали save_dir — виходило дві папки: одна порожня, одна заповнена.
    save_dir = os.path.join(PROJECT_DIR, EXPERIMENT_NAME)
    split = VALIDATION_CONFIG.get("split", "val")
    imgsz = VALIDATION_CONFIG.get("imgsz", 640)
    img_paths, _ = _load_yolo_val_data(YAML_PATH, split)

    results = validate_model(model_path=model_path, **kwargs)
    save_dir = str(getattr(results, "save_dir", None) or save_dir)

    # GT з .txt → COCO JSON (pycocotools без цього не працює)
    anno_val_path = os.path.join(save_dir, "annotations_val.json")
    yolo_split_to_coco_annotations(YAML_PATH, anno_val_path, split=split, imgsz=imgsz)
    # predictions.json валідатор вже пише при save_json (engine/validator.py); добираємо лише якщо файлу немає
    _ensure_predictions_coco_json(results, save_dir, img_paths, imgsz)

    metrics = extract_metrics(results)
    coco_metrics = run_coco_eval_metrics(save_dir, YAML_PATH, annotation_path=anno_val_path)
    defaults = {"ap_by_size": metrics.get("ap_by_size"), "ap_by_class_area": []}
    for k in ("ap_by_size", "ar_maxdets1", "ar_maxdets10", "ar_maxdets100", "ar_small", "ar_medium", "ar_large", "ap_by_class_area"):
        metrics[k] = coco_metrics.get(k, defaults.get(k, 0.0))

    # Збереження результатів (у ту ж директорію, що й валідатор)
    if save_results_flag:
        save_results(results, metrics, model_path=model_path, output_dir=save_dir)
    
    return results, metrics


if __name__ == "__main__":
    main()
