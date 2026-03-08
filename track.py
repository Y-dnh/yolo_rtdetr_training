"""
Модуль для обробки відео: детекція обраною моделлю кожні N фреймів + трекінг NanoTrack,
збереження вихідного відео з накладеними боксами та ID треків.
Усі параметри конфігурації знаходяться на початку файлу.

Перемикач MODEL_TYPE дозволяє обрати архітектуру:
  - "yolo"   -> ultralytics.YOLO
  - "rtdetr" -> ultralytics.RTDETR

USE_SAHI=True: Slicing Aided Hyper Inference — нарізка кадру на перекриваючі фрагменти
  для кращого виявлення дрібних об'єктів у великих кадрах.

BENCHMARK_MODE=True: детальний профайлінг (час кожної фази, FPS), звіт *_benchmark.txt
"""

import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Tuple
from numpy._core.numeric import True_
import torch

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
# ВИБІР АРХІТЕКТУРИ ДЕТЕКЦИЇ
# =============================================================================
# Визначає, яка модель використовується для детекції об'єктів на кадрі.
VALID_MODEL_TYPES = {"yolo", "rtdetr"}
MODEL_TYPE = "yolo"   # "yolo" — ultralytics.YOLO, "rtdetr" — ultralytics.RTDETR

# Ключі конфігу інференсу, які застосовуються тільки для YOLO (при "rtdetr" їх викидають).
YOLO_ONLY_INFERENCE_KEYS = {"agnostic_nms"}  # class-agnostic NMS; RT-DETR не використовує NMS
RTDETR_ONLY_INFERENCE_KEYS: set[str] = set()

# =============================================================================
# БАЗОВА КОНФІГУРАЦІЯ: ШЛЯХИ
# =============================================================================
PROJECT_NAME = "yolo26m"
RUNS_DIR = os.path.join(BASE_DIR, "runs")
# У WSL задай: export YOLO_DATASET_ROOT=/mnt/d/dataset_for_training
DATASET_ROOT = os.environ.get("YOLO_DATASET_ROOT", "D:/dataset_for_training")
PROJECT_DIR = os.path.join(RUNS_DIR, PROJECT_NAME)
YAML_PATH = os.path.join(DATASET_ROOT, "data.yaml")
MODEL_PATH = os.path.join(PROJECT_DIR, "baseline", "weights", "best.pt")   

# Вхідне відео або папка з відео для трекінгу.
# Якщо вказана папка — опрацьовуються всі відеофайли у ній (рекурсивно не шукаємо).
# Вихід: tracked_videos/<назва_моделі>/<ім'я_відео>_tracked.mp4 та .txt з логами
VIDEO_INPUT_PATH = "D:/work/diff_stuff/test_videos/test2.mp4"

# Розширення файлів, що вважаються відео (при вказівці папки).
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".flv"}

# Benchmark: True = профайлінг (заміри по фазах, звіт _benchmark.txt)
BENCHMARK_MODE = True
BENCHMARK_CUDA_SYNC = True
BENCHMARK_WRITE_VIDEO = True   # False = тільки профайлінг, без запису відео
BENCHMARK_MAX_FRAMES = None    # None = все відео

# Як часто запускати детекцію: модель працює тільки на кадрах 1, 1+N, 1+2N, ...; між ними лише NanoTrack.
DETECTION_INTERVAL = 5

# =============================================================================
# ПАРАМЕТРИ ІНФЕРЕНСУ (model.predict)
# =============================================================================
INFERENCE_CONFIG = {
    "conf": 0.25,            # мінімальний confidence детекції (нижче — відкидається)
    "iou": 0.3,              # IoU поріг для NMS (об'єднання дублікатів боксів)
    "imgsz": (1920, 1088),    # розмір зображення на вході моделі (краще як у навчанні)
    "max_det": 300,          # максимум детекцій на один кадр
    "half": False,           # FP16 інференс (швидше на GPU)
    "device": 0,             # 0 = CUDA GPU (було None — падало на CPU)
    "verbose": False,
    "agnostic_nms": False,   # [тільки YOLO] NMS без урахування класу
    "classes": None,         # фільтр класів (None = усі класи)
}

# =============================================================================
# SAHI (Slicing Aided Hyper Inference)
# =============================================================================
# Розбиває зображення на перекриваючі фрагменти, запускає детекцію на кожному,
# та об'єднує результати. Ефективно для виявлення дрібних об'єктів у великих кадрах.
USE_SAHI = False                          # True = увімкнути SAHI, False = звичайна детекція

SAHI_SLICE_WIDTH = 1024                    # ширина фрагменту (px)
SAHI_SLICE_HEIGHT = 1024                   # висота фрагменту (px)
SAHI_OVERLAP_WIDTH_RATIO = 0.2            # перекриття по ширині (0.0–1.0)
SAHI_OVERLAP_HEIGHT_RATIO = 0.2           # перекриття по висоті (0.0–1.0)
SAHI_PERFORM_STANDARD_PRED = True         # додатково запустити детекцію на повному кадрі
SAHI_POSTPROCESS_TYPE = "NMS"             # "NMS" або "NMM" (Non-Maximum Merging)
SAHI_POSTPROCESS_MATCH_METRIC = "IOU"     # "IOU" або "IOS" (Intersection over Smaller)
SAHI_POSTPROCESS_MATCH_THRESHOLD = 0.5    # поріг IoU/IoS для злиття дублікатів між фрагментами
SAHI_POSTPROCESS_CLASS_AGNOSTIC = False   # True = злиття без урахування класу


# =============================================================================
# NANOTRACK: ШЛЯХИ ДО ONNX-МОДЕЛЕЙ
# =============================================================================
# --- NanoTrack: ONNX-моделі (v2 або v3) ---
NANOTRACK_VERSION = "v2"  # "v2" або "v3"
NANOTRACK_DIR = os.path.join(BASE_DIR, "nanotrack")
if NANOTRACK_VERSION == "v3":
    NANOTRACK_BACKBONE = os.path.join(NANOTRACK_DIR, "v3", "nanotrack_backbone.onnx")
    NANOTRACK_NECKHEAD = os.path.join(NANOTRACK_DIR, "v3", "nanotrack_head.onnx")
else:
    NANOTRACK_BACKBONE = os.path.join(NANOTRACK_DIR, "v2", "nanotrack_backbone_sim.onnx")
    NANOTRACK_NECKHEAD = os.path.join(NANOTRACK_DIR, "v2", "nanotrack_head_sim.onnx")


# =============================================================================
# ПАРАМЕТРИ NANOTRACKER (життя треків, злиття, ReID)
# =============================================================================
# Не керують тим, коли запускається детекція — лише тим, як довго живуть треки та коли їх показувати.

MAX_AGE = 20        # скільки кадрів трек може жити без оновлення детекцією; після цього видаляється
MIN_HITS = 2        # мінімум попадань детекції по треку, щоб трек почали показувати (фільтр шуму)
IOU_THRESHOLD = 0.25 # мінімальний IoU між боксом детекції та треком, щоб вважати їх одним об'єктом
CONFIRM_THRESHOLD = 2   # після скількох попадань трек вважається «підтвердженим»
MIN_SEC_STABLE = 0.25    # мінімальний час (сек) у полі зору, щоб трек став «стабільним»

# Оптичний потік: передбачення руху треків між кадрами (швидше/точніше за багато треків).
USE_OPTICAL_FLOW_PREDICT = True
OPTICAL_FLOW_THRESHOLD = 8   # від якої кількості треків увімкнути optical flow замість оновлення кожного
ADAPTIVE_UPDATE = True       # адаптивно перемикатися на optical flow при великій кількості треків
ADAPTIVE_THRESHOLD = 10      # поріг кількості треків для адаптивного режиму

# ReID: відновлення втрачених треків (наприклад, після перекриття) за зовнішнім виглядом та позицією.
ENABLE_REID = True
REID_BUFFER_TIME = 20.0           # скільки секунд зберігати «втрачені» треки для пошуку збігу
REID_IOU_THRESHOLD = 0.15         # IoU поріг для кандидатів ReID
REID_APPEARANCE_THRESHOLD = 0.5   # поріг схожості зовнішнього вигляду (0–1)
REID_POSITION_WEIGHT = 0.4        # вага позиції у скорі ReID
REID_APPEARANCE_WEIGHT = 0.4      # вага зовнішнього вигляду
REID_SIZE_WEIGHT = 0.2           # вага розміру боксу
REID_MIN_TRACK_QUALITY = 5       # мінімальна «якість» треку (наприклад, hit_streak), щоб його зберігати в ReID-буфері


# =============================================================================
# ВІЗУАЛІЗАЦІЯ: КЛАСИ ТА КОЛЬОРИ (BGR)
# =============================================================================
CLASS_NAMES = {0: "person", 1: "car", 2: "truck"}
CLASS_COLORS = [
    (0, 0, 255),    # 0 — person, червоний
    (255, 0, 0),    # 1 — car, синій
    (0, 255, 0),    # 2 — truck, зелений
]
# Стиль візуалізації як у visualization.py: подвійна рамка, напівпрозорий фон мітки, HERSHEY_DUPLEX
VIS_FONT = cv2.FONT_HERSHEY_DUPLEX
VIS_TEXT_SCALE = 0.6
VIS_TEXT_THICKNESS = 2
VIS_TEXT_COLOR = (255, 255, 255)
VIS_LABEL_PADDING = 8
VIS_LINE_HEIGHT = 25
VIS_LABEL_MARGIN = 5
VIS_LABEL_BG_ALPHA = 0.8       # прозорість фону мітки (0–1)
VIS_BBOX_THICKNESS = 3          # товщина основної рамки
VIS_BBOX_INNER_THICKNESS = 1    # товщина внутрішнього «підсвіту»


# =============================================================================
# ФУНКЦІЇ
# =============================================================================
def validate_model_type() -> None:
    if MODEL_TYPE not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Невідомий MODEL_TYPE: '{MODEL_TYPE}'. Допустимі: {sorted(VALID_MODEL_TYPES)}"
        )


def collect_videos_from_folder(folder_path: str) -> list[str]:
    """
    Повертає список шляхів до відеофайлів у вказаній папці (тільки один рівень, без підпапок).
    Відфільтровано по VIDEO_EXTENSIONS, відсортовано за іменем.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    videos = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(str(p.resolve()))
    return sorted(videos)


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


def _run_detection_standard(model, frame: np.ndarray, frame_w: int, frame_h: int) -> list:
    """
    Стандартна детекція на повному кадрі. Повертає список dict з ключами
    'box' (cx, cy, w, h у 0–1) та 'cls_id'.
    """
    cfg = get_inference_config()
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


# ----------------------------- SAHI helpers ----------------------------------

def _sahi_generate_slices(
    img_w: int, img_h: int,
    slice_w: int, slice_h: int,
    overlap_w_ratio: float, overlap_h_ratio: float,
) -> list[Tuple[int, int, int, int]]:
    """Повертає список (x1, y1, x2, y2) координат перекриваючих фрагментів."""
    step_x = max(1, int(slice_w * (1 - overlap_w_ratio)))
    step_y = max(1, int(slice_h * (1 - overlap_h_ratio)))
    slices = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x + slice_w, img_w)
            y2 = min(y + slice_h, img_h)
            x1 = max(0, x2 - slice_w)
            y1 = max(0, y2 - slice_h)
            if (x1, y1, x2, y2) not in slices:
                slices.append((x1, y1, x2, y2))
            if x2 >= img_w:
                break
            x += step_x
        if y2 >= img_h:
            break
        y += step_y
    return slices


def _sahi_compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """IoU між двома боксами (x1, y1, x2, y2) у абсолютних координатах."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _sahi_compute_ios(box_a: Tuple, box_b: Tuple) -> float:
    """IoS (Intersection over Smaller) між двома боксами (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _sahi_merge_detections(
    detections: list,
    match_threshold: float,
    match_metric: str = "IOU",
    postprocess_type: str = "NMS",
    class_agnostic: bool = False,
) -> list:
    """
    Злиття детекцій з перекриваючих фрагментів.
    NMS — класичне Non-Maximum Suppression (залишає найкращий).
    NMM — Non-Maximum Merging (зважене середнє координат).
    """
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    match_fn = _sahi_compute_ios if match_metric.upper() == "IOS" else _sahi_compute_iou

    keep: list = []
    used = [False] * len(detections)

    for i in range(len(detections)):
        if used[i]:
            continue
        best = detections[i]
        used[i] = True
        matched_boxes = [best["xyxy"]]
        matched_confs = [best["conf"]]

        for j in range(i + 1, len(detections)):
            if used[j]:
                continue
            if not class_agnostic and detections[j]["cls_id"] != best["cls_id"]:
                continue
            score = match_fn(best["xyxy"], detections[j]["xyxy"])
            if score >= match_threshold:
                used[j] = True
                matched_boxes.append(detections[j]["xyxy"])
                matched_confs.append(detections[j]["conf"])

        if postprocess_type.upper() == "NMM" and len(matched_boxes) > 1:
            total_c = sum(matched_confs)
            avg_x1 = sum(b[0] * c for b, c in zip(matched_boxes, matched_confs)) / total_c
            avg_y1 = sum(b[1] * c for b, c in zip(matched_boxes, matched_confs)) / total_c
            avg_x2 = sum(b[2] * c for b, c in zip(matched_boxes, matched_confs)) / total_c
            avg_y2 = sum(b[3] * c for b, c in zip(matched_boxes, matched_confs)) / total_c
            best = {**best, "xyxy": (avg_x1, avg_y1, avg_x2, avg_y2), "conf": max(matched_confs)}

        keep.append(best)

    return keep


def _run_detection_sahi(model, frame: np.ndarray, frame_w: int, frame_h: int) -> list:
    """
    SAHI-детекція: нарізка кадру на фрагменти → інференс на кожному →
    (опціонально) інференс на повному кадрі → злиття результатів.
    """
    cfg = get_inference_config()
    cfg = {k: v for k, v in cfg.items() if v is not None}

    slices = _sahi_generate_slices(
        frame_w, frame_h,
        SAHI_SLICE_WIDTH, SAHI_SLICE_HEIGHT,
        SAHI_OVERLAP_WIDTH_RATIO, SAHI_OVERLAP_HEIGHT_RATIO,
    )

    raw: list = []

    for (sx1, sy1, sx2, sy2) in slices:
        patch = frame[sy1:sy2, sx1:sx2]
        if patch.size == 0:
            continue
        results = model.predict(patch, **cfg)
        for result in results:
            for box in result.boxes:
                bx1, by1, bx2, by2 = map(float, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0]) if box.conf is not None and len(box.conf) else 0.0
                raw.append({
                    "xyxy": (bx1 + sx1, by1 + sy1, bx2 + sx1, by2 + sy1),
                    "cls_id": cls_id,
                    "conf": conf,
                })

    if SAHI_PERFORM_STANDARD_PRED:
        results = model.predict(frame, **cfg)
        for result in results:
            for box in result.boxes:
                bx1, by1, bx2, by2 = map(float, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0]) if box.conf is not None and len(box.conf) else 0.0
                raw.append({
                    "xyxy": (bx1, by1, bx2, by2),
                    "cls_id": cls_id,
                    "conf": conf,
                })

    merged = _sahi_merge_detections(
        raw,
        match_threshold=SAHI_POSTPROCESS_MATCH_THRESHOLD,
        match_metric=SAHI_POSTPROCESS_MATCH_METRIC,
        postprocess_type=SAHI_POSTPROCESS_TYPE,
        class_agnostic=SAHI_POSTPROCESS_CLASS_AGNOSTIC,
    )

    detections = []
    for d in merged:
        x1, y1, x2, y2 = d["xyxy"]
        w = (x2 - x1) / frame_w
        h = (y2 - y1) / frame_h
        cx = (x1 + x2) / 2.0 / frame_w
        cy = (y1 + y2) / 2.0 / frame_h
        detections.append({"box": (cx, cy, w, h), "cls_id": d["cls_id"], "conf": d["conf"]})
    return detections


def run_detection(model, frame: np.ndarray, frame_w: int, frame_h: int) -> list:
    """
    Запуск детекції на кадрі. При USE_SAHI=True використовує SAHI (нарізка на фрагменти),
    інакше — звичайний інференс на повному кадрі.
    Повертає список dict з ключами 'box' (cx, cy, w, h у 0–1), 'cls_id', 'conf'.
    """
    if USE_SAHI:
        return _run_detection_sahi(model, frame, frame_w, frame_h)
    return _run_detection_standard(model, frame, frame_w, frame_h)


def _vis_get_text_size(text: str) -> Tuple[int, int]:
    """Розмір тексту в стилі visualization.py."""
    return cv2.getTextSize(text, VIS_FONT, VIS_TEXT_SCALE, VIS_TEXT_THICKNESS)[0]


def _vis_draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = VIS_LABEL_BG_ALPHA,
) -> None:
    """Текст з напівпрозорим фоном та обводкою (стиль visualization.py)."""
    x, y = position
    tw, th = _vis_get_text_size(text)
    bg_x1 = max(0, x - VIS_LABEL_PADDING // 2)
    bg_y1 = max(0, y - th - VIS_LABEL_PADDING)
    bg_x2 = min(frame.shape[1], x + tw + VIS_LABEL_PADDING // 2)
    bg_y2 = min(frame.shape[0], y + VIS_LABEL_PADDING // 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), tuple(max(0, c - 50) for c in color), 1)
    cv2.putText(frame, text, (x, y - VIS_LABEL_PADDING // 4), VIS_FONT, VIS_TEXT_SCALE, VIS_TEXT_COLOR, VIS_TEXT_THICKNESS)


def _vis_draw_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]) -> None:
    """Подвійна рамка: основна + внутрішній підсвіт (стиль visualization.py)."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, VIS_BBOX_THICKNESS)
    inner = tuple(min(255, c + 30) for c in color)
    cv2.rectangle(frame, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), inner, VIS_BBOX_INNER_THICKNESS)


def draw_tracks(frame: np.ndarray, tracked: list, class_names: dict, colors: list) -> None:
    """
    Малювання боксів та міток у стилі visualization.py: подвійна рамка, напівпрозорий фон мітки,
    білий текст (HERSHEY_DUPLEX), два рядки — клас (confidence) та ID.
    """
    h, w = frame.shape[:2]
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

        # Порядок як у visualization.py: спочатку клас (conf), потім ID
        texts = [f"{cls_name} ({conf_str})", f"ID: {track_id}"]

        _vis_draw_bbox(frame, x1, y1, x2, y2, color)

        total_height = sum(_vis_get_text_size(t)[1] + VIS_LABEL_PADDING for t in texts)
        start_y = y1 - VIS_LABEL_MARGIN
        if start_y - total_height < 0:
            current_y = y1 + VIS_LINE_HEIGHT
        else:
            current_y = start_y

        for text in texts:
            tw, th = _vis_get_text_size(text)
            text_x = max(VIS_LABEL_PADDING, min(x1, w - tw - VIS_LABEL_PADDING))
            text_y = max(th + VIS_LABEL_PADDING, min(current_y, h - VIS_LABEL_PADDING))
            _vis_draw_text_with_background(frame, text, (text_x, text_y), color)
            if start_y - total_height < 0:
                current_y += VIS_LINE_HEIGHT
            else:
                current_y -= th + VIS_LABEL_PADDING


# =============================================================================
# BENCHMARK (профайлінг пайплайну)
# =============================================================================
def _benchmark_sync():
    """Синхронізація CUDA для точного виміру GPU операцій (лише при BENCHMARK_CUDA_SYNC)."""
    if BENCHMARK_CUDA_SYNC and torch.cuda.is_available():
        torch.cuda.synchronize()


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f}"


def _fmt_pct(part: float, total: float) -> str:
    if total <= 0:
        return "0.0%"
    return f"{part / total * 100:.1f}%"


class BenchmarkStats:
    """Збирач статистики по фазах для benchmark."""

    def __init__(self):
        self.frame_read_times = []
        self.detection_times = []
        self.tracker_update_times = []
        self.drawing_times = []
        self.frame_write_times = []
        self.detection_frame_count = 0
        self.tracking_only_frame_count = 0
        self.total_detections = 0
        self.total_tracks_drawn = 0
        self.warmup_time = 0.0
        self.model_load_time = 0.0

    def add_frame(self, read_t, detect_t, tracker_t, draw_t, write_t,
                  is_detection_frame, num_detections, num_tracks):
        self.frame_read_times.append(read_t)
        self.detection_times.append(detect_t)
        self.tracker_update_times.append(tracker_t)
        self.drawing_times.append(draw_t)
        self.frame_write_times.append(write_t)
        if is_detection_frame:
            self.detection_frame_count += 1
        else:
            self.tracking_only_frame_count += 1
        self.total_detections += num_detections
        self.total_tracks_drawn += num_tracks

    @property
    def n(self):
        return len(self.frame_read_times)


def _generate_benchmark_report(stats: BenchmarkStats, pipeline_total: float,
                               model_path: str, video_path: str,
                               device_name: str, gpu_name: str,
                               width: int, height: int, fps_video: float,
                               total_frames: int, cfg: dict, has_tracker: bool) -> str:
    """Генерація текстового звіту benchmark."""
    n = stats.n
    if n == 0:
        return "No frames processed.\n"

    phases = {
        "Frame Read (I/O)": stats.frame_read_times,
        "Detection (GPU)": stats.detection_times,
        "Tracker (CPU)": stats.tracker_update_times,
        "Drawing (CPU)": stats.drawing_times,
        "Frame Write (I/O)": stats.frame_write_times,
    }
    phase_totals = {name: sum(arr) for name, arr in phases.items()}
    measured_total = sum(phase_totals.values())
    overhead = pipeline_total - measured_total
    effective_fps = n / pipeline_total if pipeline_total > 0 else 0
    det_only = [stats.detection_times[i] for i in range(n) if stats.detection_times[i] > 0]
    track_only_times = [stats.tracker_update_times[i] for i in range(n) if stats.detection_times[i] == 0]
    track_with_det_times = [stats.tracker_update_times[i] for i in range(n) if stats.detection_times[i] > 0]

    lines = [
        "=" * 72,
        "  BENCHMARK REPORT: TRACKING PIPELINE PROFILING",
        "=" * 72,
        f"  Date:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Model:            {Path(model_path).name}",
        f"  Architecture:     {MODEL_TYPE.upper()}",
        f"  Device:           {device_name}",
        f"  GPU:              {gpu_name}",
        f"  Video:            {video_path}",
        f"  Resolution:       {width}x{height} @ {fps_video:.1f} FPS",
        f"  Frames processed: {n} / {total_frames}",
        f"  Detection every:  {DETECTION_INTERVAL} frames",
        f"  Tracker:          {'NanoTrack v' + NANOTRACK_VERSION if has_tracker else 'None'}",
        f"  SAHI:             {'ON' if USE_SAHI else 'OFF'}",
        f"  CUDA Sync:        {BENCHMARK_CUDA_SYNC}",
    ]
    if USE_SAHI:
        n_slices = len(_sahi_generate_slices(width, height, SAHI_SLICE_WIDTH, SAHI_SLICE_HEIGHT,
                                             SAHI_OVERLAP_WIDTH_RATIO, SAHI_OVERLAP_HEIGHT_RATIO))
        lines.extend([
            "",
            f"  SAHI config:",
            f"    Slice size:      {SAHI_SLICE_WIDTH}x{SAHI_SLICE_HEIGHT}",
            f"    Overlap:         {SAHI_OVERLAP_WIDTH_RATIO:.0%} x {SAHI_OVERLAP_HEIGHT_RATIO:.0%}",
            f"    Slices count:    {n_slices}",
            f"    Full-frame pred: {SAHI_PERFORM_STANDARD_PRED}",
            f"    Postprocess:     {SAHI_POSTPROCESS_TYPE} ({SAHI_POSTPROCESS_MATCH_METRIC}, thr={SAHI_POSTPROCESS_MATCH_THRESHOLD})",
            f"    Class agnostic:  {SAHI_POSTPROCESS_CLASS_AGNOSTIC}",
        ])
    lines.extend([
        "",
        "  Inference config:",
    ])
    for k in ["imgsz", "conf", "iou", "half", "max_det", "device"]:
        if k in cfg:
            lines.append(f"    {k:16s} = {cfg[k]}")
    lines.extend([
        "",
        "-" * 72, "  TOTAL PIPELINE", "-" * 72,
        f"  Wall-clock time:    {pipeline_total:.3f} s",
        f"  Measured phases:    {measured_total:.3f} s",
        f"  Overhead (loop):    {overhead:.3f} s ({_fmt_pct(overhead, pipeline_total)})",
        f"  Effective FPS:      {effective_fps:.1f}",
        f"  Avg frame time:     {_fmt_ms(pipeline_total / n)} ms",
        "",
        "-" * 72, "  TIME BREAKDOWN BY PHASE", "-" * 72,
        f"  {'Phase':<24s} {'Total (s)':>10s} {'% of total':>10s} {'Avg (ms)':>10s} {'Min (ms)':>10s} {'Max (ms)':>10s}",
        "  " + "-" * 68,
    ])
    for name, arr in phases.items():
        t_total = sum(arr)
        t_avg = t_total / n if n > 0 else 0
        t_min = min(arr) if arr else 0
        t_max = max(arr) if arr else 0
        lines.append(f"  {name:<24s} {t_total:>10.3f} {_fmt_pct(t_total, pipeline_total):>10s} {_fmt_ms(t_avg):>10s} {_fmt_ms(t_min):>10s} {_fmt_ms(t_max):>10s}")
    lines.extend([
        "  " + "-" * 68,
        f"  {'SUM (measured)':<24s} {measured_total:>10.3f} {_fmt_pct(measured_total, pipeline_total):>10s} {_fmt_ms(measured_total / n):>10s}",
        "",
        "-" * 72, "  DETECTION FRAMES (GPU inference)", "-" * 72,
        f"  Detection frames:     {stats.detection_frame_count} / {n}",
        f"  Tracking-only frames: {stats.tracking_only_frame_count} / {n}",
        f"  Total detections:     {stats.total_detections}",
    ])
    if stats.detection_frame_count > 0:
        lines.append(f"  Avg detections/frame: {stats.total_detections / stats.detection_frame_count:.1f}")
    if det_only:
        det_total = phase_totals["Detection (GPU)"]
        det_fps = stats.detection_frame_count / det_total if det_total > 0 else 0
        lines.extend([
            f"  Detection time (det frames only):",
            f"    Avg: {_fmt_ms(sum(det_only) / len(det_only))} ms",
            f"    Min: {_fmt_ms(min(det_only))} ms",
            f"    Max: {_fmt_ms(max(det_only))} ms",
            f"  Detection speed:      {det_fps:.1f} inf/s  (інференсів на секунду)",
        ])
    lines.extend([
        "",
        "-" * 72, "  TRACKER UPDATE (CPU)", "-" * 72,
        f"  Total tracks drawn:   {stats.total_tracks_drawn}",
    ])
    if n > 0:
        lines.append(f"  Avg tracks/frame:     {stats.total_tracks_drawn / n:.1f}")
    track_total = phase_totals["Tracker (CPU)"]
    track_avg_ms = (track_total / n * 1000) if n > 0 and track_total > 0 else 0
    track_fps = 1000 / track_avg_ms if track_avg_ms > 0 else 0
    if track_with_det_times:
        lines.extend([
            f"  Tracker on det frames:",
            f"    Avg: {_fmt_ms(sum(track_with_det_times) / len(track_with_det_times))} ms",
        ])
    if track_only_times:
        lines.extend([
            f"  Tracker on non-det frames:",
            f"    Avg: {_fmt_ms(sum(track_only_times) / len(track_only_times))} ms",
        ])
    if track_total > 0:
        lines.append(f"  Tracker speed:        {track_avg_ms:.1f} ms/frame  ({track_fps:.0f} frames/s по трекеру)")
    io_total = phase_totals["Frame Read (I/O)"] + phase_totals["Frame Write (I/O)"]
    compute_total = phase_totals["Detection (GPU)"] + phase_totals["Tracker (CPU)"]
    compute_fps = n / compute_total if compute_total > 0 else 0

    lines.extend([
        "",
        "-" * 72, "  I/O", "-" * 72,
        f"  Total I/O time:   {io_total:.3f} s ({_fmt_pct(io_total, pipeline_total)})",
        "",
        "-" * 72, "  INITIALIZATION", "-" * 72,
        f"  Model load:       {_fmt_ms(stats.model_load_time)} ms",
        f"  Warmup:           {_fmt_ms(stats.warmup_time)} ms",
        "",
        "-" * 72, "  SUMMARY", "-" * 72,
        f"  Effective FPS (pipeline):   {effective_fps:.1f}  (з I/O та малюванням)",
        f"  Compute FPS (det+track):    {compute_fps:.1f}  (тільки детекція + трекінг, без I/O)",
        f"  CUDA Sync:                  {BENCHMARK_CUDA_SYNC}",
        "",
        "=" * 72,
    ])
    return "\n".join(lines)


def run_tracking(
    video_input_path: str,
    model_path: str = MODEL_PATH,
    detection_interval: int = DETECTION_INTERVAL,
    benchmark_mode: bool = False,
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
    stem = Path(model_path).stem
    model_name = PROJECT_NAME if stem == PROJECT_NAME else f"{PROJECT_NAME}_{stem}"
    output_dir = os.path.join(BASE_DIR, "tracked_videos", model_name)
    os.makedirs(output_dir, exist_ok=True)
    video_stem = Path(video_input_path).stem
    output_path = os.path.join(output_dir, f"{video_stem}_tracked.mp4")
    log_path = os.path.join(output_dir, f"{video_stem}_tracked.txt")
    benchmark_log_path = os.path.join(output_dir, f"{video_stem}_benchmark.txt")

    benchmark_stats = BenchmarkStats() if benchmark_mode else None

    # use_nano = False  # BENCHMARK: вимкнено NanoTrack
    use_nano = os.path.isfile(NANOTRACK_BACKBONE) and os.path.isfile(NANOTRACK_NECKHEAD)
    if not use_nano:
        print(
            "Увага: моделі NanoTrack не знайдено. "
            "Буде використано режим лише детекції кожні N кадрів (без трекінгу між кадрами)."
        )

    print(f"Завантаження моделі: {Path(model_path).name}")
    t0 = time.perf_counter()
    model = load_model(model_path)
    if benchmark_stats is not None:
        benchmark_stats.model_load_time = time.perf_counter() - t0
    class_names = getattr(model, "names", None) or CLASS_NAMES

    # Warmup: переносить модель на цільовий девайс і прогріває
    cfg_warmup = get_inference_config()
    cfg_warmup = {k: v for k, v in cfg_warmup.items() if v is not None}
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    if benchmark_stats is not None:
        _benchmark_sync()
        t0 = time.perf_counter()
    model.predict(dummy, **{**cfg_warmup, "verbose": False, "imgsz": 64})
    if benchmark_stats is not None:
        _benchmark_sync()
        benchmark_stats.warmup_time = time.perf_counter() - t0

    # Визначення девайсу моделі (після warmup — вже на цільовому девайсі)
    gpu_name = "N/A"
    try:
        device = next(model.model.parameters()).device
        device_name = f"{device}"
        if device.type == "cuda":
            import torch
            gpu_name = torch.cuda.get_device_name(device.index)
            device_name = f"CUDA:{device.index} ({gpu_name})"
    except Exception:
        dev_cfg = INFERENCE_CONFIG.get("device")
        device_name = f"CUDA:{dev_cfg} (TensorRT)" if dev_cfg is not None else "TensorRT (GPU)"

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"Помилка: не вдалося відкрити відео: {video_input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames_to_process = total_frames
    if benchmark_mode and BENCHMARK_MAX_FRAMES:
        frames_to_process = min(total_frames, BENCHMARK_MAX_FRAMES)

    print()
    print("=" * 60)
    print("BENCHMARK: TRACKING PIPELINE" if benchmark_mode else "TRACK VIDEO")
    print("=" * 60)
    print(f"Відео: {video_input_path}")
    print(f"Вихід: {output_path}")
    print(f"Модель: {MODEL_PATH}")
    print(f"Архітектура: {MODEL_TYPE.upper()}")
    print(f"Девайс: {device_name}")
    cfg = get_inference_config()
    print(f"Кадрів: {total_frames}, {fps:.1f} FPS, {width}x{height}")
    print(f"Розмір інференсу: {cfg.get('imgsz')}x{cfg.get('imgsz')}")
    print(f"Детекція кожні: {detection_interval} фреймів")
    print(f"Інференс: conf={cfg.get('conf')}, iou={cfg.get('iou')}, imgsz={cfg.get('imgsz')}, max_det={cfg.get('max_det')}, half={cfg.get('half')}")
    if USE_SAHI:
        n_slices = len(_sahi_generate_slices(width, height, SAHI_SLICE_WIDTH, SAHI_SLICE_HEIGHT,
                                             SAHI_OVERLAP_WIDTH_RATIO, SAHI_OVERLAP_HEIGHT_RATIO))
        print(f"SAHI: ON  |  вікно={SAHI_SLICE_WIDTH}x{SAHI_SLICE_HEIGHT}, "
              f"перекриття={SAHI_OVERLAP_WIDTH_RATIO:.0%}x{SAHI_OVERLAP_HEIGHT_RATIO:.0%}, "
              f"фрагментів={n_slices}, повний_кадр={SAHI_PERFORM_STANDARD_PRED}, "
              f"злиття={SAHI_POSTPROCESS_TYPE}({SAHI_POSTPROCESS_MATCH_METRIC}, "
              f"thr={SAHI_POSTPROCESS_MATCH_THRESHOLD})")
    else:
        print("SAHI: OFF")
    if benchmark_mode:
        print(f"Benchmark: CUDA_SYNC={BENCHMARK_CUDA_SYNC}, WRITE_VIDEO={BENCHMARK_WRITE_VIDEO}, MAX_FRAMES={BENCHMARK_MAX_FRAMES}")
    print("=" * 60)
    print()

    writer = None
    if not benchmark_mode or BENCHMARK_WRITE_VIDEO:
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
    detection_counts = defaultdict(int)   # cls_id -> кількість детекцій
    track_durations = {}                  # (track_id, cls_id) -> тривалість (с)
    start_time = time.perf_counter()
    pbar = tqdm(total=frames_to_process if frames_to_process else None, unit="frame",
                desc="Benchmark" if benchmark_mode else "Track")
    try:
        while True:
            if benchmark_mode and frames_to_process and frame_counter >= frames_to_process:
                break
            t_read_s = time.perf_counter() if benchmark_stats else 0
            ret, frame = cap.read()
            t_read = time.perf_counter() - t_read_s if benchmark_stats else 0
            if not ret or frame is None:
                break
            frame_counter += 1
            pbar.update(1)
            frame_h, frame_w = frame.shape[:2]

            is_det_frame = (frame_counter % detection_interval == 1 or frame_counter == 1)
            num_detections = 0
            t_detect = 0.0

            if is_det_frame:
                if benchmark_stats:
                    _benchmark_sync()
                    t_det_s = time.perf_counter()
                detections = run_detection(model, frame, frame_w, frame_h)
                if benchmark_stats:
                    _benchmark_sync()
                    t_detect = time.perf_counter() - t_det_s
                num_detections = len(detections)
                for d in detections:
                    cid = d.get("cls_id")
                    if cid is not None:
                        detection_counts[cid] += 1
                if benchmark_stats:
                    t_track_s = time.perf_counter()
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
                if benchmark_stats:
                    t_track = time.perf_counter() - t_track_s
            else:
                t_track = 0.0
                if benchmark_stats:
                    t_track_s = time.perf_counter()
                if tracker is not None:
                    try:
                        last_tracked = tracker.update(None, frame)
                    except Exception:
                        pass
                if benchmark_stats:
                    t_track = time.perf_counter() - t_track_s

            num_tracks = len(last_tracked)
            if benchmark_stats:
                t_draw_s = time.perf_counter()
            for obj in last_tracked:
                cid = getattr(obj, "cls_id", None)
                if cid is None:
                    cid = 0
                tid = getattr(obj, "track_id", "")
                if tid:
                    first = getattr(obj, "first_seen", None)
                    last = getattr(obj, "last_seen", None)
                    if first is not None and last is not None:
                        track_durations[(tid, cid)] = last - first

            draw_tracks(frame, last_tracked, class_names, CLASS_COLORS)
            if benchmark_stats:
                t_draw = time.perf_counter() - t_draw_s
            if writer is not None:
                if benchmark_stats:
                    t_write_s = time.perf_counter()
                writer.write(frame)
                if benchmark_stats:
                    t_write = time.perf_counter() - t_write_s
            else:
                t_write = 0.0 if benchmark_stats else 0
            if benchmark_stats:
                benchmark_stats.add_frame(
                    t_read, t_detect, t_track, t_draw, t_write,
                    is_det_frame, num_detections, num_tracks,
                )
    except Exception as e:
        print(f"Помилка під час обробки: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()

    elapsed_sec = time.perf_counter() - start_time
    fps_processed = frame_counter / elapsed_sec if elapsed_sec > 0 else 0.0

    # Статистика по класах: детекції, треки, середня тривалість треку
    all_cls_ids = sorted(set(detection_counts.keys()) | {cid for (_, cid) in track_durations})
    tracks_per_class = defaultdict(int)
    duration_sum_per_class = defaultdict(float)
    for (tid, cid), dur in track_durations.items():
        tracks_per_class[cid] += 1
        duration_sum_per_class[cid] += dur
    avg_duration_per_class = {}
    for cid in all_cls_ids:
        n = tracks_per_class.get(cid, 0)
        if n > 0:
            avg_duration_per_class[cid] = duration_sum_per_class[cid] / n
        else:
            avg_duration_per_class[cid] = None

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
        "--- СТАТИСТИКА ПО КЛАСАХ ---",
    ]
    for cid in all_cls_ids:
        name = class_names.get(cid, f"cls_{cid}")
        det_count = detection_counts.get(cid, 0)
        tr_count = tracks_per_class.get(cid, 0)
        avg_dur = avg_duration_per_class.get(cid)
        avg_dur_str = f"{avg_dur:.2f} с" if avg_dur is not None else "—"
        log_lines.append(f"  {name} (id={cid}): детекцій={det_count}, треків={tr_count}, середня тривалість треку={avg_dur_str}")
    log_lines.extend([
        "",
        "--- МОДЕЛЬ ДЕТЕКЦІЇ ---",
        f"Модель: {model_path}",
        f"Архітектура: {MODEL_TYPE}",
        "",
        "--- КОНФІГ ІНФЕРЕНСУ (model.predict) ---",
    ])
    for k, v in sorted(cfg.items()):
        log_lines.append(f"  {k}: {v}")
    log_lines.extend([
        "",
        "--- ДЕТЕКЦІЯ ---",
        f"  DETECTION_INTERVAL: {detection_interval}  # кожні N фреймів",
        "",
        "--- SAHI (Slicing Aided Hyper Inference) ---",
        f"  USE_SAHI: {USE_SAHI}",
    ])
    if USE_SAHI:
        n_slices = len(_sahi_generate_slices(width, height, SAHI_SLICE_WIDTH, SAHI_SLICE_HEIGHT,
                                             SAHI_OVERLAP_WIDTH_RATIO, SAHI_OVERLAP_HEIGHT_RATIO))
        log_lines.extend([
            f"  SAHI_SLICE_WIDTH: {SAHI_SLICE_WIDTH}",
            f"  SAHI_SLICE_HEIGHT: {SAHI_SLICE_HEIGHT}",
            f"  SAHI_OVERLAP_WIDTH_RATIO: {SAHI_OVERLAP_WIDTH_RATIO}",
            f"  SAHI_OVERLAP_HEIGHT_RATIO: {SAHI_OVERLAP_HEIGHT_RATIO}",
            f"  SAHI_PERFORM_STANDARD_PRED: {SAHI_PERFORM_STANDARD_PRED}",
            f"  SAHI_POSTPROCESS_TYPE: {SAHI_POSTPROCESS_TYPE}",
            f"  SAHI_POSTPROCESS_MATCH_METRIC: {SAHI_POSTPROCESS_MATCH_METRIC}",
            f"  SAHI_POSTPROCESS_MATCH_THRESHOLD: {SAHI_POSTPROCESS_MATCH_THRESHOLD}",
            f"  SAHI_POSTPROCESS_CLASS_AGNOSTIC: {SAHI_POSTPROCESS_CLASS_AGNOSTIC}",
            f"  Кількість фрагментів: {n_slices}",
        ])
    log_lines.extend([
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
        f"  VIS_BBOX_THICKNESS: {VIS_BBOX_THICKNESS}",
        f"  VIS_TEXT_SCALE: {VIS_TEXT_SCALE}",
        f"  VIS_TEXT_THICKNESS: {VIS_TEXT_THICKNESS}",
        "=" * 60,
    ])
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"Лог збережено: {log_path}")
    except Exception as e:
        print(f"Не вдалося записати лог: {e}")

    if benchmark_mode and benchmark_stats is not None and benchmark_stats.n > 0:
        benchmark_report = _generate_benchmark_report(
            benchmark_stats, elapsed_sec, model_path, video_input_path,
            device_name, gpu_name, width, height, fps,
            frames_to_process, cfg, tracker is not None,
        )
        try:
            with open(benchmark_log_path, "w", encoding="utf-8") as f:
                f.write(benchmark_report)
            print(f"Benchmark звіт збережено: {benchmark_log_path}")
            print(f"\n{benchmark_report}")
        except Exception as e:
            print(f"Не вдалося записати benchmark звіт: {e}")

    print()
    done_parts = [f"Лог: {log_path}"]
    if benchmark_mode and benchmark_stats is not None and benchmark_stats.n > 0:
        done_parts.append(f"Benchmark: {benchmark_log_path}")
    if writer:
        done_parts.append(f"Відео: {output_path}")
    print("Готово. " + ", ".join(done_parts))
    return output_path if writer else log_path


def main():
    """Головна функція для запуску обробки відео або всіх відео у папці."""
    input_path = VIDEO_INPUT_PATH.strip()
    if not input_path:
        print("Задайте VIDEO_INPUT_PATH у конфігу на початку файлу.")
        return None

    path = Path(input_path)
    if path.is_dir():
        video_paths = collect_videos_from_folder(input_path)
        if not video_paths:
            print(f"У папці не знайдено відео (розширення: {', '.join(sorted(VIDEO_EXTENSIONS))}): {input_path}")
            return None
        print(f"Знайдено відео у папці: {len(video_paths)}")
        results = []
        for i, video_path in enumerate(video_paths, 1):
            print(f"\n[{i}/{len(video_paths)}] Обробка: {Path(video_path).name}")
            out = run_tracking(video_path, MODEL_PATH, DETECTION_INTERVAL, benchmark_mode=BENCHMARK_MODE)
            if out:
                results.append(out)
        print(f"\nОпрацьовано: {len(results)}/{len(video_paths)} відео.")
        return results
    else:
        return run_tracking(input_path, MODEL_PATH, DETECTION_INTERVAL, benchmark_mode=BENCHMARK_MODE)


if __name__ == "__main__":
    main()
