"""
Модуль для обробки відео: детекція обраною моделлю кожні N фреймів + трекінг NanoTrack,
збереження вихідного відео з накладеними боксами та ID треків.
Усі параметри конфігурації знаходяться на початку файлу.

Перемикач MODEL_TYPE дозволяє обрати архітектуру:
  - "yolo"   -> ultralytics.YOLO
  - "rtdetr" -> ultralytics.RTDETR
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Фікс для правильного відображення tqdm у Windows PowerShell
if sys.platform == "win32":
    os.system("")  # Включає ANSI escape sequences підтримку

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO, RTDETR

# Додаємо корінь проєкту в шлях для імпорту tracking
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tracking import NanoTracker, TrackedObject


# =============================================================================
# ВИБІР АРХІТЕКТУРИ: "yolo" або "rtdetr"
# =============================================================================
VALID_MODEL_TYPES = {"yolo", "rtdetr"}
MODEL_TYPE = "yolo"        # <-- ПЕРЕМИКАЧ: "yolo" або "rtdetr"

# =============================================================================
# ПАРАМЕТРИ, СПЕЦИФІЧНІ ДЛЯ КОЖНОЇ АРХІТЕКТУРИ (інференс/predict)
# =============================================================================
# Ключі, які є ТІЛЬКИ у YOLO (видаляються при MODEL_TYPE="rtdetr")
YOLO_ONLY_INFERENCE_KEYS = {
    "agnostic_nms",  # Class-agnostic NMS — RT-DETR не використовує NMS
}
RTDETR_ONLY_INFERENCE_KEYS: set[str] = set()

# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ
# =============================================================================
PROJECT_NAME = "yolov8x-p2_for_autolabelling"
PROJECT_DIR = os.path.join(BASE_DIR, "old_dataset_runs", PROJECT_NAME)
MODEL_PATH = os.path.join(PROJECT_DIR, "baseline", "weights", "best.pt")

# --- Відео ---
# Вихід зберігається в tracked_videos/<назва_моделі>/<ім'я_відео>_tracked.mp4 та .txt з логами
VIDEO_INPUT_PATH = "E:/DPSU/dataset_videos/uzhorod/videos_to_extract/006_02.12.2025_08.20_08.40.mkv"

# --- Детекція: кожні N фреймів запускати модель; між ними — трекінг NanoTrack ---
DETECTION_INTERVAL = 10

# =============================================================================
# ПАРАМЕТРИ ІНФЕРЕНСУ (передаються в model.predict(), як у validate.py)
# =============================================================================
INFERENCE_CONFIG = {
    # Пороги детекції
    "conf": 0.25,            # Confidence threshold
    "iou": 0.5,              # NMS IoU threshold
    "imgsz": 960,           # Розмір входу (має відповідати навчанню/експорту)
    "max_det": 300,          # Макс. кількість детекцій на кадр
    # Точність та пристрій
    "half": True,           # FP16 інференс (швидше на GPU, перевірити сумісність)
    "device": None,          # None = авто (cuda якщо є)
    "verbose": False,
    # [YOLO-only] RT-DETR не використовує NMS
    "agnostic_nms": False,
    "classes": None,         # Фільтр класів (None = усі)
}

# --- NanoTrack: ONNX-моделі (nanotrack_backbone_sim.onnx, nanotrack_head_sim.onnx) ---
NANOTRACK_DIR = os.path.join(BASE_DIR, "nanotrack")
NANOTRACK_BACKBONE = os.path.join(NANOTRACK_DIR, "v2", "nanotrack_backbone_sim.onnx")
NANOTRACK_NECKHEAD = os.path.join(NANOTRACK_DIR, "v2", "nanotrack_head_sim.onnx")

# --- Класи та кольори (BGR): person червоний, car синій, truck зелений ---
CLASS_NAMES = {0: "person", 1: "car", 2: "truck"}
CLASS_COLORS = [
    (0, 0, 255),    # person — червоний
    (255, 0, 0),    # car — синій
    (0, 255, 0),    # truck — зелений
]
BBOX_THICKNESS = 1
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
LABEL_PADDING = 4

# --- Параметри трекінгу (NanoTracker) ---
MAX_AGE = 10
MIN_HITS = 2
IOU_THRESHOLD = 0.3
CONFIRM_THRESHOLD = 5
MIN_SEC_STABLE = 1.0
USE_OPTICAL_FLOW_PREDICT = True
OPTICAL_FLOW_THRESHOLD = 8
ADAPTIVE_UPDATE = True
ADAPTIVE_THRESHOLD = 10
ENABLE_REID = True
REID_BUFFER_TIME = 20.0
REID_IOU_THRESHOLD = 0.15
REID_APPEARANCE_THRESHOLD = 0.5
REID_POSITION_WEIGHT = 0.4
REID_APPEARANCE_WEIGHT = 0.4
REID_SIZE_WEIGHT = 0.2
REID_MIN_TRACK_QUALITY = 5


# =============================================================================
# ФУНКЦІЇ
# =============================================================================
def validate_model_type() -> None:
    if MODEL_TYPE not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Невідомий MODEL_TYPE: '{MODEL_TYPE}'. Допустимі: {sorted(VALID_MODEL_TYPES)}"
        )


def load_model(model_path: str):
    """Завантаження YOLO або RT-DETR моделі."""
    validate_model_type()
    if MODEL_TYPE == "rtdetr" or "rtdetr" in model_path.lower():
        return RTDETR(model_path)
    return YOLO(model_path)


def filter_config(config: dict, excluded_keys: set) -> dict:
    """Фільтрує конфіг: видаляє ключі, несумісні з поточним MODEL_TYPE."""
    return {k: v for k, v in config.items() if k not in excluded_keys}


def get_inference_config(**kwargs) -> dict:
    """Повертає відфільтрований конфіг інференсу для поточного MODEL_TYPE."""
    config = {**INFERENCE_CONFIG, **kwargs}
    if MODEL_TYPE == "rtdetr":
        return filter_config(config, YOLO_ONLY_INFERENCE_KEYS)
    return filter_config(config, RTDETR_ONLY_INFERENCE_KEYS)


def run_detection(model, frame: np.ndarray, frame_w: int, frame_h: int) -> list:
    """
    Запуск детекції на кадрі. Повертає список dict з ключами 'box' (cx, cy, w, h у 0–1)
    та 'cls_id'. Використовує INFERENCE_CONFIG.
    """
    cfg = get_inference_config()
    # Прибираємо None (model.predict не приймає частину ключів як None)
    cfg = {k: v for k, v in cfg.items() if v is not None}
    results = model.predict(frame, **cfg)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0]) if box.conf is not None and len(box.conf) else 0.0
            w = (x2 - x1) / frame_w
            h = (y2 - y1) / frame_h
            cx = (x1 + x2) / 2.0 / frame_w
            cy = (y1 + y2) / 2.0 / frame_h
            detections.append({"box": (cx, cy, w, h), "cls_id": cls_id, "conf": conf})
    return detections


def draw_tracks(frame: np.ndarray, tracked: list, class_names: dict, colors: list) -> None:
    """
    Малювання боксів та міток у стилі як на зразку: тонка рамка кольору класу,
    мітка над боксом — два рядки (ID та клас з confidence), суцільний фон кольору класу, білий текст.
    """
    h, w = frame.shape[:2]
    white = (255, 255, 255)
    for obj in tracked:
        bbox = getattr(obj, "bbox", None)
        if bbox is None or len(bbox) != 4:
            continue
        x1_n, y1_n, x2_n, y2_n = bbox
        x1 = int(x1_n * w)
        y1 = int(y1_n * h)
        x2 = int(x2_n * w)
        y2 = int(y2_n * h)
        cls_id = getattr(obj, "cls_id", None)
        cls_id = cls_id if cls_id is not None else 0
        color = colors[cls_id % len(colors)] if colors else (0, 0, 255)
        cls_name = class_names.get(cls_id, f"cls_{cls_id}")
        track_id = getattr(obj, "track_id", "")
        confidence = getattr(obj, "confidence", None)
        conf_str = f"{confidence:.2f}" if confidence is not None else "—"

        line1 = f"ID: {track_id}"
        line2 = f"{cls_name} ({conf_str})"

        (w1, h1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)
        (w2, h2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)
        label_w = max(w1, w2) + LABEL_PADDING * 2
        label_h = h1 + h2 + LABEL_PADDING * 2
        spacing = 2
        ty = max(0, y1 - label_h)
        tx = x1
        cv2.rectangle(frame, (tx, ty), (tx + label_w, ty + label_h), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)
        cv2.putText(
            frame, line1,
            (tx + LABEL_PADDING, ty + LABEL_PADDING + h1),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            white,
            TEXT_THICKNESS,
        )
        cv2.putText(
            frame, line2,
            (tx + LABEL_PADDING, ty + LABEL_PADDING + h1 + spacing + h2),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            white,
            TEXT_THICKNESS,
        )


def run_tracking(
    video_input_path: str,
    model_path: str = MODEL_PATH,
    detection_interval: int = DETECTION_INTERVAL,
) -> str | None:
    """
    Головний пайплайн: відкрити відео, детекція кожні detection_interval кадрів,
    трекінг NanoTrack на кожному кадрі, малювання боксів, запис у tracked_videos/<модель>/.
    Поруч із відео створюється .txt з логами (час, FPS, роздільність тощо).

    Returns:
        Шлях до збереженого відео або None при помилці.
    """
    if not video_input_path or not os.path.isfile(video_input_path):
        print(f"Помилка: не знайдено відео: {video_input_path}")
        return None
    if not os.path.isfile(model_path):
        print(f"Помилка: не знайдено модель: {model_path}")
        return None

    # Вихід: tracked_videos/<назва_моделі>/<ім'я_відео>_tracked.mp4 та .txt
    model_name = f"{PROJECT_NAME}_{Path(model_path).stem}"
    output_dir = os.path.join(BASE_DIR, "tracked_videos", model_name)
    os.makedirs(output_dir, exist_ok=True)
    video_stem = Path(video_input_path).stem
    output_path = os.path.join(output_dir, f"{video_stem}_tracked.mp4")
    log_path = os.path.join(output_dir, f"{video_stem}_tracked.txt")

    use_nano = os.path.isfile(NANOTRACK_BACKBONE) and os.path.isfile(NANOTRACK_NECKHEAD)
    if not use_nano:
        print(
            "Увага: моделі NanoTrack не знайдено. "
            "Буде використано режим лише детекції кожні N кадрів (без трекінгу між кадрами)."
        )

    print(f"Завантаження моделі: {Path(model_path).name}")
    model = load_model(model_path)
    class_names = getattr(model, "names", None) or CLASS_NAMES

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"Помилка: не вдалося відкрити відео: {video_input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    print()
    print("=" * 60)
    print("TRACK VIDEO")
    print("=" * 60)
    print(f"Відео: {video_input_path}")
    print(f"Вихід: {output_path}")
    print(f"Модель: {Path(model_path).name}")
    print(f"Архітектура: {MODEL_TYPE.upper()}")
    cfg = get_inference_config()
    print(f"Кадрів: {total_frames}, {fps:.1f} FPS, {width}x{height}")
    print(f"Детекція кожні: {detection_interval} фреймів")
    print(f"Інференс: conf={cfg.get('conf')}, iou={cfg.get('iou')}, imgsz={cfg.get('imgsz')}, max_det={cfg.get('max_det')}, half={cfg.get('half')}")
    print("=" * 60)
    print()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Помилка: не вдалося створити вихідний файл: {output_path}")
        cap.release()
        return None

    tracker = None
    try:
        tracker = NanoTracker(
            class_names=class_names,
            backbone_path=NANOTRACK_BACKBONE,
            neckhead_path=NANOTRACK_NECKHEAD,
            max_age=MAX_AGE,
            min_hits=MIN_HITS,
            iou_threshold=IOU_THRESHOLD,
            confirm_threshold=CONFIRM_THRESHOLD,
            min_sec_stable=MIN_SEC_STABLE,
            use_optical_flow_predict=USE_OPTICAL_FLOW_PREDICT,
            optical_flow_threshold=OPTICAL_FLOW_THRESHOLD,
            adaptive_update=ADAPTIVE_UPDATE,
            adaptive_threshold=ADAPTIVE_THRESHOLD,
            enable_reid=ENABLE_REID,
            reid_buffer_time=REID_BUFFER_TIME,
            reid_iou_threshold=REID_IOU_THRESHOLD,
            reid_appearance_threshold=REID_APPEARANCE_THRESHOLD,
            reid_position_weight=REID_POSITION_WEIGHT,
            reid_appearance_weight=REID_APPEARANCE_WEIGHT,
            reid_size_weight=REID_SIZE_WEIGHT,
            reid_min_track_quality=REID_MIN_TRACK_QUALITY,
        )
    except Exception as e:
        print(f"Не вдалося створити NanoTracker (потрібні ONNX та OpenCV з TrackerNano): {e}")
        tracker = None

    frame_counter = 0
    last_tracked = []
    start_time = time.perf_counter()
    pbar = tqdm(total=total_frames if total_frames else None, unit="frame", desc="Track")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_counter += 1
            pbar.update(1)
            frame_h, frame_w = frame.shape[:2]

            if frame_counter % detection_interval == 1 or frame_counter == 1:
                detections = run_detection(model, frame, frame_w, frame_h)
                if tracker is not None:
                    try:
                        last_tracked = tracker.update(detections, frame)
                    except Exception as e:
                        print(f"Помилка трекера (кадр {frame_counter}): {e}")
                        last_tracked = []
                else:
                    # Режим без трекера: малюємо лише поточні детекції (норм. bbox)
                    last_tracked = []
                    for d in detections:
                        cx, cy, w, h = d["box"]
                        bbox_n = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                        last_tracked.append(
                            type("Obj", (), {
                                "bbox": bbox_n,
                                "cls_id": d.get("cls_id"),
                                "track_id": "",
                                "confidence": d.get("conf", 0.0),
                            })()
                        )
            else:
                if tracker is not None:
                    try:
                        last_tracked = tracker.update(None, frame)
                    except Exception:
                        pass
                # Якщо трекера немає, last_tracked залишається з попереднього кадру (або порожній)

            draw_tracks(frame, last_tracked, class_names, CLASS_COLORS)
            writer.write(frame)
    except Exception as e:
        print(f"Помилка під час обробки: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
        cap.release()
        writer.release()

    elapsed_sec = time.perf_counter() - start_time
    fps_processed = frame_counter / elapsed_sec if elapsed_sec > 0 else 0.0

    cfg = get_inference_config()
    log_lines = [
        "=" * 60,
        "РЕЗУЛЬТАТИ ОПРАЦЮВАННЯ",
        "=" * 60,
        f"Дата/час: {datetime.now().isoformat()}",
        f"Вхідне відео: {video_input_path}",
        f"Вихідне відео: {output_path}",
        f"Роздільність: {width}x{height}",
        f"Розширення виходу: mp4 (codec mp4v)",
        f"Кадрів у джерелі: {total_frames}",
        f"Оброблено кадрів: {frame_counter}",
        f"Час опрацювання (с): {elapsed_sec:.2f}",
        f"FPS при обробці: {fps_processed:.2f}",
        "",
        "--- МОДЕЛЬ ДЕТЕКЦІЇ ---",
        f"Модель: {model_path}",
        f"Архітектура: {MODEL_TYPE}",
        "",
        "--- КОНФІГ ІНФЕРЕНСУ (model.predict) ---",
    ]
    for k, v in sorted(cfg.items()):
        log_lines.append(f"  {k}: {v}")
    log_lines.extend([
        "",
        "--- ДЕТЕКЦІЯ ---",
        f"  DETECTION_INTERVAL: {detection_interval}  # кожні N фреймів",
        "",
        "--- NANOTRACK ---",
        f"  NANOTRACK_BACKBONE: {NANOTRACK_BACKBONE}",
        f"  NANOTRACK_NECKHEAD: {NANOTRACK_NECKHEAD}",
        "",
        "--- ПАРАМЕТРИ ТРЕКИНГУ (NanoTracker) ---",
        f"  MAX_AGE: {MAX_AGE}",
        f"  MIN_HITS: {MIN_HITS}",
        f"  IOU_THRESHOLD: {IOU_THRESHOLD}",
        f"  CONFIRM_THRESHOLD: {CONFIRM_THRESHOLD}",
        f"  MIN_SEC_STABLE: {MIN_SEC_STABLE}",
        f"  USE_OPTICAL_FLOW_PREDICT: {USE_OPTICAL_FLOW_PREDICT}",
        f"  OPTICAL_FLOW_THRESHOLD: {OPTICAL_FLOW_THRESHOLD}",
        f"  ADAPTIVE_UPDATE: {ADAPTIVE_UPDATE}",
        f"  ADAPTIVE_THRESHOLD: {ADAPTIVE_THRESHOLD}",
        f"  ENABLE_REID: {ENABLE_REID}",
        f"  REID_BUFFER_TIME: {REID_BUFFER_TIME}",
        f"  REID_IOU_THRESHOLD: {REID_IOU_THRESHOLD}",
        f"  REID_APPEARANCE_THRESHOLD: {REID_APPEARANCE_THRESHOLD}",
        f"  REID_POSITION_WEIGHT: {REID_POSITION_WEIGHT}",
        f"  REID_APPEARANCE_WEIGHT: {REID_APPEARANCE_WEIGHT}",
        f"  REID_SIZE_WEIGHT: {REID_SIZE_WEIGHT}",
        f"  REID_MIN_TRACK_QUALITY: {REID_MIN_TRACK_QUALITY}",
        "",
        "--- КЛАСИ ---",
        f"  CLASS_NAMES: {CLASS_NAMES}",
        "",
        "--- ВІЗУАЛІЗАЦІЯ ---",
        f"  BBOX_THICKNESS: {BBOX_THICKNESS}",
        f"  TEXT_SCALE: {TEXT_SCALE}",
        f"  TEXT_THICKNESS: {TEXT_THICKNESS}",
        "=" * 60,
    ])
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"Лог збережено: {log_path}")
    except Exception as e:
        print(f"Не вдалося записати лог: {e}")

    print()
    print("Готово. Вихідне відео: " + output_path)
    return output_path


def main():
    """Головна функція для запуску обробки відео."""
    video_path = VIDEO_INPUT_PATH.strip()
    if not video_path:
        print("Задайте VIDEO_INPUT_PATH у конфігу на початку файлу.")
        return None
    return run_tracking(video_path, MODEL_PATH, DETECTION_INTERVAL)


if __name__ == "__main__":
    main()
