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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Tuple

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
PROJECT_NAME = "yolo26s_pretrained"
PROJECT_DIR = os.path.join(BASE_DIR, PROJECT_NAME)
MODEL_PATH = os.path.join(PROJECT_DIR, "baseline", "weights", "best.pt")   # TensorRT FP16 модель

# Вхідне відео або папка з відео для трекінгу.
# Якщо вказана папка — опрацьовуються всі відеофайли у ній (рекурсивно не шукаємо).
# Вихід: tracked_videos/<назва_моделі>/<ім'я_відео>_tracked.mp4 та .txt з логами.
VIDEO_INPUT_PATH = "D:/videos_for_test"

# Розширення файлів, що вважаються відео (при вказівці папки).
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".flv"}

# Як часто запускати детекцію: модель працює тільки на кадрах 1, 1+N, 1+2N, ...; між ними лише NanoTrack.
DETECTION_INTERVAL = 10   # BENCHMARK: детекція кожен кадр

# =============================================================================
# ПАРАМЕТРИ ІНФЕРЕНСУ (model.predict)
# =============================================================================
INFERENCE_CONFIG = {
    "conf": 0.25,            # мінімальний confidence детекції (нижче — відкидається)
    "iou": 0.5,              # IoU поріг для NMS (об'єднання дублікатів боксів)
    "imgsz": 1024,            # розмір зображення на вході моделі (краще як у навчанні)
    "max_det": 300,          # максимум детекцій на один кадр
    "half": True,            # FP16 інференс (швидше на GPU)
    "device": 0,             # 0 = CUDA GPU (було None — падало на CPU)
    "verbose": False,
    "agnostic_nms": False,   # [тільки YOLO] NMS без урахування класу
    "classes": None,         # фільтр класів (None = усі класи)
}

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

MAX_AGE = 10        # скільки кадрів трек може жити без оновлення детекцією; після цього видаляється
MIN_HITS = 2        # мінімум попадань детекції по треку, щоб трек почали показувати (фільтр шуму)
IOU_THRESHOLD = 0.3 # мінімальний IoU між боксом детекції та треком, щоб вважати їх одним об'єктом
CONFIRM_THRESHOLD = 5   # після скількох попадань трек вважається «підтвердженим»
MIN_SEC_STABLE = 1.0    # мінімальний час (сек) у полі зору, щоб трек став «стабільним»

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

    # use_nano = False  # BENCHMARK: вимкнено NanoTrack
    use_nano = os.path.isfile(NANOTRACK_BACKBONE) and os.path.isfile(NANOTRACK_NECKHEAD)
    if not use_nano:
        print(
            "Увага: моделі NanoTrack не знайдено. "
            "Буде використано режим лише детекції кожні N кадрів (без трекінгу між кадрами)."
        )

    print(f"Завантаження моделі: {Path(model_path).name}")
    model = load_model(model_path)
    class_names = getattr(model, "names", None) or CLASS_NAMES

    # Warmup: переносить модель на цільовий девайс і прогріває
    cfg_warmup = get_inference_config()
    cfg_warmup = {k: v for k, v in cfg_warmup.items() if v is not None}
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    model.predict(dummy, **{**cfg_warmup, "verbose": False, "imgsz": 64})

    # Визначення девайсу моделі (після warmup — вже на цільовому девайсі)
    try:
        device = next(model.model.parameters()).device
        device_name = f"{device}"
        if device.type == "cuda":
            import torch
            device_name = f"CUDA:{device.index} ({torch.cuda.get_device_name(device.index)})"
    except Exception:
        # TensorRT .engine — визначаємо з конфігу
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

    print()
    print("=" * 60)
    print("TRACK VIDEO")
    print("=" * 60)
    print(f"Відео: {video_input_path}")
    print(f"Вихід: {output_path}")
    print(f"Модель: {Path(model_path).name}")
    print(f"Архітектура: {MODEL_TYPE.upper()}")
    print(f"Девайс: {device_name}")
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
    detection_counts = defaultdict(int)   # cls_id -> кількість детекцій
    track_durations = {}                  # (track_id, cls_id) -> тривалість (с)
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
                for d in detections:
                    cid = d.get("cls_id")
                    if cid is not None:
                        detection_counts[cid] += 1
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

    print()
    print("Готово. Вихідне відео: " + output_path)
    return output_path


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
            out = run_tracking(video_path, MODEL_PATH, DETECTION_INTERVAL)
            if out:
                results.append(out)
        print(f"\nОпрацьовано: {len(results)}/{len(video_paths)} відео.")
        return results
    else:
        return run_tracking(input_path, MODEL_PATH, DETECTION_INTERVAL)


if __name__ == "__main__":
    main()
