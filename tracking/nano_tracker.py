"""
NanoTracker: детекції кожні N фреймів + OpenCV TrackerNano між ними.
Інтерфейс: update(detections, frame) — detections=None для кадрів без детекції.
"""

import logging
import threading
import time
from collections import namedtuple
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2 as cv
import numpy as np

from tracking.reid_manager import ReIDManager, get_reid_manager

logger = logging.getLogger(__name__)

TrackedObject = namedtuple(
    "TrackedObject",
    [
        "track_id", "cls_id", "cls_name", "stable", "announced",
        "age", "hit_streak", "first_seen", "last_seen",
        "bbox", "cx", "cy", "w", "h", "device_id",
        "is_new", "confidence",
    ],
)


def _iou_xyxy(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)


def _nms_detections(detections: List[dict], iou_threshold: float = 0.5) -> List[dict]:
    if len(detections) <= 1:
        return detections
    sorted_dets = sorted(enumerate(detections), key=lambda x: x[1].get("conf", 1.0), reverse=True)
    keep = []
    suppressed = set()
    for orig_idx, det in sorted_dets:
        if orig_idx in suppressed:
            continue
        keep.append(det)
        cx1, cy1, w1, h1 = det["box"]
        for other_idx, other_det in sorted_dets:
            if other_idx in suppressed or other_idx == orig_idx:
                continue
            cx2, cy2, w2, h2 = other_det["box"]
            box1 = (cx1 - w1 / 2, cy1 - h1 / 2, cx1 + w1 / 2, cy1 + h1 / 2)
            box2 = (cx2 - w2 / 2, cy2 - h2 / 2, cx2 + w2 / 2, cy2 + h2 / 2)
            if _iou_xyxy(box1, box2) > iou_threshold:
                suppressed.add(other_idx)
    return keep


def _create_cv_nano_tracker(backbone_path: str, neckhead_path: str):
    params = cv.TrackerNano_Params()
    params.backbone = backbone_path
    params.neckhead = neckhead_path
    return cv.TrackerNano_create(params)


class _NanoInternalTrack:
    def __init__(
        self,
        frame: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
        cls_id: Optional[int] = None,
        device_id: Optional[str] = None,
        min_sec_stable: float = 2.0,
        track_id: Optional[str] = None,
        first_seen: Optional[float] = None,
        total_hits: int = 0,
        backbone_path: str = "",
        neckhead_path: str = "",
        was_reidentified: bool = False,
        conf: float = 0.0,
    ):
        # track_id передається з NanoTracker (числовий рядок "1", "2", ...) або при ReID (старий id)
        self.track_id = track_id if track_id is not None else "0"
        self.cls_id = cls_id
        self.conf = conf  # останній confidence з детекції (для візуалізації)
        self.device_id = device_id
        self.time_since_update = 0
        self.hit_streak = total_hits
        self.age = 0
        self.first_seen = first_seen if first_seen else time.time()
        self.last_seen = time.time()
        self.stable = False
        self.announced = False
        self.MIN_DETECTION_DURATION = min_sec_stable
        self.bbox_xyxy = bbox_xyxy
        self.prev_bbox_xyxy = None
        self._state_norm = (0.0, 0.0, 0.0, 0.0)
        self.has_been_emitted = False
        self.was_reidentified = was_reidentified
        self.tracker = _create_cv_nano_tracker(backbone_path, neckhead_path)
        x1, y1, x2, y2 = bbox_xyxy
        self.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        h, w = frame.shape[:2]
        self._update_state_norm(w, h)

    def predict(self, frame: np.ndarray):
        ok, box = self.tracker.update(frame)
        self.age += 1
        self.time_since_update += 1
        if ok:
            x, y, w, h = box
            self.bbox_xyxy = (int(x), int(y), int(x + w), int(y + h))
            h_frame, w_frame = frame.shape[:2]
            self._update_state_norm(w_frame, h_frame)
            if not self.stable and (self.last_seen - self.first_seen) >= self.MIN_DETECTION_DURATION:
                self.stable = True
                self.announced = True

    def _update_state_norm(self, frame_w: int, frame_h: int):
        x1, y1, x2, y2 = self.bbox_xyxy
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        self._state_norm = (
            cx / frame_w if frame_w else 0.0,
            cy / frame_h if frame_h else 0.0,
            w / frame_w if frame_w else 0.0,
            h / frame_h if frame_h else 0.0,
        )

    def get_state(self):
        return self._state_norm


class NanoTracker:
    def __init__(
        self,
        max_age: int = 10,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
        confirm_threshold: int = 5,
        min_sec_stable: float = 1.0,
        class_names: Optional[Dict[int, str]] = None,
        device_id: Optional[str] = None,
        use_optical_flow_predict: bool = True,
        optical_flow_threshold: int = 8,
        adaptive_update: bool = True,
        adaptive_threshold: int = 10,
        enable_reid: bool = True,
        reid_buffer_time: float = 20.0,
        reid_iou_threshold: float = 0.15,
        reid_appearance_threshold: float = 0.5,
        reid_position_weight: float = 0.4,
        reid_appearance_weight: float = 0.4,
        reid_size_weight: float = 0.2,
        reid_min_track_quality: int = 5,
        backbone_path: str = "",
        neckhead_path: str = "",
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.confirm_threshold = confirm_threshold
        self.min_sec_stable = min_sec_stable
        self.class_names = class_names or {}
        self.device_id = device_id
        self.tracks: List[_NanoInternalTrack] = []
        self.frame_size: Tuple[int, int] = (0, 0)
        self.use_optical_flow_predict = use_optical_flow_predict
        self.optical_flow_threshold = optical_flow_threshold
        self.adaptive_update = adaptive_update
        self.adaptive_threshold = adaptive_threshold
        self.frame_counter = 0
        self.prev_gray = None
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 0.01),
        )
        self.enable_reid = enable_reid
        self.reid_manager: Optional[ReIDManager] = None
        if enable_reid:
            self.reid_manager = get_reid_manager(
                reid_buffer_time=reid_buffer_time,
                reid_iou_threshold=reid_iou_threshold,
                reid_appearance_threshold=reid_appearance_threshold,
                reid_position_weight=reid_position_weight,
                reid_appearance_weight=reid_appearance_weight,
                reid_size_weight=reid_size_weight,
                min_track_quality=reid_min_track_quality,
            )
        self.track_nms_threshold = 0.5
        self._backbone_path = backbone_path
        self._neckhead_path = neckhead_path
        self._next_track_id = 1  # Прості числові ID: 1, 2, 3, ...

    def _predict_with_optical_flow(self, frame: np.ndarray, adaptive: bool = False):
        if not self.tracks:
            return
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_h, frame_w = frame.shape[:2]
        if self.prev_gray is None:
            self.prev_gray = gray
            return
        prev_points = []
        track_indices = []
        for i, tr in enumerate(self.tracks):
            if adaptive and len(self.tracks) >= self.adaptive_threshold:
                if (self.frame_counter + i) % 2 != 0:
                    tr.age += 1
                    tr.time_since_update += 1
                    continue
            x1, y1, x2, y2 = tr.bbox_xyxy
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            prev_points.append([cx, cy])
            track_indices.append(i)
        if not prev_points:
            self.prev_gray = gray
            return
        prev_points = np.array(prev_points, dtype=np.float32).reshape(-1, 1, 2)
        next_points, status, _ = cv.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_points, None, **self.lk_params
        )
        for i, (track_idx, st) in enumerate(zip(track_indices, status)):
            tr = self.tracks[track_idx]
            tr.age += 1
            tr.time_since_update += 1
            if st[0] == 1:
                old_cx, old_cy = prev_points[i][0]
                new_cx, new_cy = next_points[i][0]
                dx, dy = new_cx - old_cx, new_cy - old_cy
                x1, y1, x2, y2 = tr.bbox_xyxy
                tr.bbox_xyxy = (
                    int(max(0, x1 + dx)),
                    int(max(0, y1 + dy)),
                    int(min(frame_w, x2 + dx)),
                    int(min(frame_h, y2 + dy)),
                )
                tr._update_state_norm(frame_w, frame_h)
                if not tr.stable and (time.time() - tr.first_seen) >= tr.MIN_DETECTION_DURATION:
                    tr.stable = True
                    tr.announced = True
        self.prev_gray = gray

    def _add_track_to_reid_buffer(self, tr: _NanoInternalTrack, frame: np.ndarray):
        if not self.enable_reid or self.reid_manager is None:
            return
        frame_h, frame_w = frame.shape[:2]
        cx_n, cy_n, w_n, h_n = tr.get_state()
        self.reid_manager.add_lost_track(
            track_id=tr.track_id,
            cls_id=tr.cls_id,
            device_id=tr.device_id,
            bbox_xyxy=tr.bbox_xyxy,
            bbox_norm=(cx_n, cy_n, w_n, h_n),
            frame_size=(frame_w, frame_h),
            first_seen=tr.first_seen,
            last_seen=tr.last_seen,
            total_age=tr.age,
            total_hits=tr.hit_streak,
            was_stable=tr.stable,
        )

    def _try_reid_match(
        self,
        frame: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
        cls_id: Optional[int],
        conf: float = 0.0,
    ) -> Optional[_NanoInternalTrack]:
        if not self.enable_reid or self.reid_manager is None:
            return None
        match = self.reid_manager.find_match(
            frame=frame, bbox_xyxy=bbox_xyxy, cls_id=cls_id, device_id=self.device_id
        )
        if match is None:
            return None
        restored = _NanoInternalTrack(
            frame=frame,
            bbox_xyxy=bbox_xyxy,
            cls_id=cls_id if cls_id is not None else match.cls_id,
            device_id=self.device_id,
            min_sec_stable=self.min_sec_stable,
            track_id=match.track_id,
            first_seen=match.first_seen,
            total_hits=match.total_hits,
            backbone_path=self._backbone_path,
            neckhead_path=self._neckhead_path,
            was_reidentified=True,
            conf=conf,
        )
        if match.was_stable:
            restored.stable = True
            restored.announced = True
        restored.has_been_emitted = True
        return restored

    def _update_track_appearance(self, tr: _NanoInternalTrack, frame: np.ndarray):
        if not self.enable_reid or self.reid_manager is None:
            return
        self.reid_manager.update_track_features(
            track_id=tr.track_id,
            frame=frame,
            bbox_xyxy=tr.bbox_xyxy,
            prev_bbox_xyxy=tr.prev_bbox_xyxy,
        )
        tr.prev_bbox_xyxy = tr.bbox_xyxy

    def _remove_duplicate_tracks(self):
        if len(self.tracks) <= 1:
            return
        sorted_tracks = sorted(self.tracks, key=lambda t: (t.hit_streak, t.age), reverse=True)
        keep = []
        suppressed = set()
        for tr in sorted_tracks:
            if tr.track_id in suppressed:
                continue
            keep.append(tr)
            for other in sorted_tracks:
                if other.track_id in suppressed or other.track_id == tr.track_id:
                    continue
                if tr.cls_id is not None and other.cls_id is not None and tr.cls_id != other.cls_id:
                    continue
                if _iou_xyxy(tr.bbox_xyxy, other.bbox_xyxy) > self.track_nms_threshold:
                    suppressed.add(other.track_id)
                    if self.reid_manager:
                        self.reid_manager.remove_track(other.track_id)
        self.tracks = keep

    def _collect_tracks(self) -> List[TrackedObject]:
        result: List[TrackedObject] = []
        frame_w, frame_h = self.frame_size
        for tr in self.tracks:
            if tr.time_since_update > self.max_age or tr.hit_streak < self.min_hits:
                continue
            cx_n, cy_n, w_n, h_n = tr.get_state()
            x1_n = cx_n - w_n / 2.0
            y1_n = cy_n - h_n / 2.0
            x2_n = cx_n + w_n / 2.0
            y2_n = cy_n + h_n / 2.0
            cls_name = self.class_names.get(tr.cls_id, f"Class {tr.cls_id}") if tr.cls_id is not None else None
            is_new = not getattr(tr, "has_been_emitted", False)
            result.append(
                TrackedObject(
                    track_id=tr.track_id,
                    cls_id=tr.cls_id,
                    cls_name=cls_name,
                    stable=tr.stable,
                    announced=tr.announced,
                    age=tr.age,
                    hit_streak=tr.hit_streak,
                    first_seen=tr.first_seen,
                    last_seen=tr.last_seen,
                    bbox=(x1_n, y1_n, x2_n, y2_n),
                    cx=cx_n,
                    cy=cy_n,
                    w=w_n,
                    h=h_n,
                    device_id=tr.device_id,
                    is_new=is_new,
                    confidence=getattr(tr, "conf", 0.0),
                )
            )
            tr.has_been_emitted = True
        return result

    def update(
        self,
        detections: Optional[List[dict]],
        frame: Optional[np.ndarray] = None,
    ) -> List[TrackedObject]:
        if detections is None:
            if frame is None:
                return self._collect_tracks()
            self.frame_counter += 1
            if self.use_optical_flow_predict and len(self.tracks) >= self.optical_flow_threshold:
                adaptive = self.adaptive_update and len(self.tracks) >= self.adaptive_threshold
                self._predict_with_optical_flow(frame, adaptive=adaptive)
            else:
                for tr in self.tracks:
                    tr.predict(frame)
            for t in list(self.tracks):
                if t.time_since_update > self.max_age:
                    self._add_track_to_reid_buffer(t, frame)
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            self._remove_duplicate_tracks()
            return self._collect_tracks()

        if frame is None:
            return self._collect_tracks()

        frame_h, frame_w = frame.shape[:2]
        self.frame_size = (frame_w, frame_h)
        self.frame_counter += 1
        if self.use_optical_flow_predict:
            self.prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if self.use_optical_flow_predict and len(self.tracks) >= self.optical_flow_threshold:
            self._predict_with_optical_flow(frame, adaptive=False)
        else:
            for tr in self.tracks:
                tr.predict(frame)

        if not detections:
            return self._collect_tracks()

        detections = _nms_detections(detections, iou_threshold=0.5)
        assigned = set()
        for det in detections:
            cx, cy, w, h = det["box"]
            x1 = int((cx - w / 2) * frame_w)
            y1 = int((cy - h / 2) * frame_h)
            x2 = int((cx + w / 2) * frame_w)
            y2 = int((cy + h / 2) * frame_h)
            det_bbox = (x1, y1, x2, y2)
            cls_val = det.get("cls_id") if isinstance(det.get("cls_id"), (int, type(None))) else None
            best_iou = 0
            best_track = None
            for tr in self.tracks:
                if tr.cls_id is not None and cls_val is not None and tr.cls_id != cls_val:
                    continue
                iou_val = _iou_xyxy(tr.bbox_xyxy, det_bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_track = tr
            if best_iou >= self.iou_threshold and best_track is not None:
                best_track.prev_bbox_xyxy = best_track.bbox_xyxy
                best_track.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                best_track.bbox_xyxy = det_bbox
                best_track.cls_id = cls_val if cls_val is not None else best_track.cls_id
                best_track.conf = det.get("conf", getattr(best_track, "conf", 0.0))
                best_track.time_since_update = 0
                best_track.hit_streak += 1
                best_track.last_seen = time.time()
                assigned.add(best_track.track_id)
                self._update_track_appearance(best_track, frame)
            else:
                reid_track = self._try_reid_match(
                    frame, det_bbox, cls_val, conf=det.get("conf", 0.0)
                )
                if reid_track is not None:
                    self.tracks.append(reid_track)
                    assigned.add(reid_track.track_id)
                    self._update_track_appearance(reid_track, frame)
                else:
                    new_tr = _NanoInternalTrack(
                        frame,
                        det_bbox,
                        cls_val,
                        self.device_id,
                        self.min_sec_stable,
                        track_id=str(self._next_track_id),
                        backbone_path=self._backbone_path,
                        neckhead_path=self._neckhead_path,
                        conf=det.get("conf", 0.0),
                    )
                    self._next_track_id += 1
                    self.tracks.append(new_tr)
                    assigned.add(new_tr.track_id)
                    self._update_track_appearance(new_tr, frame)

        new_active = []
        for tr in self.tracks:
            if tr.track_id in assigned:
                new_active.append(tr)
            elif tr.time_since_update <= self.max_age:
                new_active.append(tr)
            else:
                self._add_track_to_reid_buffer(tr, frame)
                if self.reid_manager:
                    self.reid_manager.track_features.pop(tr.track_id, None)
                    self.reid_manager.velocity_history.pop(tr.track_id, None)
        self.tracks = new_active
        self._remove_duplicate_tracks()
        return self._collect_tracks()
