"""
Re-Identification для трекінгу: відновлення ID об'єктів після виходу з кадру.
Оптимізовано для thermal/IR відео (інтенсивність, розмір, позиція).
"""

import time
import logging
import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class AppearanceFeatures:
    """Ознаки зовнішності для Re-ID (інтенсивність, форма)."""
    intensity_histogram: np.ndarray = field(default_factory=lambda: np.zeros(32))
    mean_intensity: float = 0.0
    max_intensity: float = 0.0
    min_intensity: float = 0.0
    std_intensity: float = 0.0
    aspect_ratio: float = 1.0
    area: float = 0.0
    update_count: int = 0

    def update(self, new_features: "AppearanceFeatures", alpha: float = 0.7):
        if self.update_count == 0:
            self.intensity_histogram = new_features.intensity_histogram.copy()
            self.mean_intensity = new_features.mean_intensity
            self.max_intensity = new_features.max_intensity
            self.min_intensity = new_features.min_intensity
            self.std_intensity = new_features.std_intensity
            self.aspect_ratio = new_features.aspect_ratio
            self.area = new_features.area
        else:
            self.intensity_histogram = alpha * new_features.intensity_histogram + (1 - alpha) * self.intensity_histogram
            self.mean_intensity = alpha * new_features.mean_intensity + (1 - alpha) * self.mean_intensity
            self.max_intensity = alpha * new_features.max_intensity + (1 - alpha) * self.max_intensity
            self.min_intensity = alpha * new_features.min_intensity + (1 - alpha) * self.min_intensity
            self.std_intensity = alpha * new_features.std_intensity + (1 - alpha) * self.std_intensity
            self.aspect_ratio = alpha * new_features.aspect_ratio + (1 - alpha) * self.aspect_ratio
            self.area = alpha * new_features.area + (1 - alpha) * self.area
        self.update_count += 1


@dataclass
class LostTrack:
    """Втрачений трек у буфері для Re-ID."""
    track_id: str
    cls_id: Optional[int]
    device_id: Optional[str]
    last_bbox_xyxy: Tuple[int, int, int, int]
    last_bbox_norm: Tuple[float, float, float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    appearance: AppearanceFeatures = field(default_factory=AppearanceFeatures)
    lost_time: float = field(default_factory=time.time)
    first_seen: float = 0.0
    last_seen: float = 0.0
    total_age: int = 0
    total_hits: int = 0
    was_stable: bool = False
    frame_size: Tuple[int, int] = (1, 1)

    def get_predicted_position(self, frames_elapsed: int = 0) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.last_bbox_xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        vx, vy = self.velocity
        pred_cx = cx + vx * frames_elapsed
        pred_cy = cy + vy * frames_elapsed
        frame_w, frame_h = self.frame_size
        pred_cx = max(w / 2, min(frame_w - w / 2, pred_cx))
        pred_cy = max(h / 2, min(frame_h - h / 2, pred_cy))
        pred_x1 = int(pred_cx - w / 2)
        pred_y1 = int(pred_cy - h / 2)
        pred_x2 = int(pred_cx + w / 2)
        pred_y2 = int(pred_cy + h / 2)
        return (pred_x1, pred_y1, pred_x2, pred_y2)


class ReIDManager:
    def __init__(
        self,
        reid_buffer_time: float = 30.0,
        reid_iou_threshold: float = 0.15,
        reid_appearance_threshold: float = 0.6,
        reid_position_weight: float = 0.4,
        reid_appearance_weight: float = 0.4,
        reid_size_weight: float = 0.2,
        histogram_bins: int = 32,
        velocity_smoothing: float = 0.5,
        min_track_quality: int = 5,
    ):
        self.reid_buffer_time = reid_buffer_time
        self.reid_iou_threshold = reid_iou_threshold
        self.reid_appearance_threshold = reid_appearance_threshold
        self.reid_position_weight = reid_position_weight
        self.reid_appearance_weight = reid_appearance_weight
        self.reid_size_weight = reid_size_weight
        self.histogram_bins = histogram_bins
        self.velocity_smoothing = velocity_smoothing
        self.min_track_quality = min_track_quality
        self.lost_tracks: Dict[str, LostTrack] = {}
        self.track_features: Dict[str, AppearanceFeatures] = {}
        self.velocity_history: Dict[str, deque] = {}
        self.stats = {"total_reids": 0, "successful_reids": 0, "failed_reids": 0}

    def extract_appearance_features(
        self, frame: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]
    ) -> AppearanceFeatures:
        features = AppearanceFeatures()
        x1, y1, x2, y2 = bbox_xyxy
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, min(w_frame - 1, x1))
        y1 = max(0, min(h_frame - 1, y1))
        x2 = max(x1 + 1, min(w_frame, x2))
        y2 = max(y1 + 1, min(h_frame, y2))
        if frame.ndim == 3:
            roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        else:
            roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return features
        hist = cv2.calcHist([roi], [0], None, [self.histogram_bins], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-7)
        features.intensity_histogram = hist
        features.mean_intensity = float(np.mean(roi))
        features.max_intensity = float(np.max(roi))
        features.min_intensity = float(np.min(roi))
        features.std_intensity = float(np.std(roi))
        w, h = x2 - x1, y2 - y1
        features.aspect_ratio = w / (h + 1e-7)
        features.area = w * h
        return features

    def compute_appearance_similarity(
        self, features1: AppearanceFeatures, features2: AppearanceFeatures
    ) -> float:
        if features1.update_count == 0 or features2.update_count == 0:
            return 0.0
        hist_sim = cv2.compareHist(
            features1.intensity_histogram.astype(np.float32),
            features2.intensity_histogram.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
        hist_sim = 1.0 - hist_sim
        mean_diff = abs(features1.mean_intensity - features2.mean_intensity) / 255.0
        max_diff = abs(features1.max_intensity - features2.max_intensity) / 255.0
        intensity_sim = 1.0 - (mean_diff + max_diff) / 2.0
        return max(0.0, min(1.0, 0.7 * hist_sim + 0.3 * intensity_sim))

    def compute_size_similarity(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        area1, area2 = w1 * h1, w2 * h2
        if area1 == 0 or area2 == 0:
            return 0.0
        area_ratio = min(area1, area2) / max(area1, area2)
        ar1 = w1 / (h1 + 1e-7)
        ar2 = w2 / (h2 + 1e-7)
        ar_ratio = min(ar1, ar2) / max(ar1, ar2)
        return 0.6 * area_ratio + 0.4 * ar_ratio

    def compute_position_similarity(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
        frame_size: Tuple[int, int],
    ) -> float:
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        frame_w, frame_h = frame_size
        diagonal = np.sqrt(frame_w ** 2 + frame_h ** 2)
        distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        similarity = 1.0 - min(1.0, (distance / diagonal) * 2)
        return max(0.0, similarity)

    def iou_xyxy(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        return inter / (a1 + a2 - inter + 1e-7)

    def update_track_features(
        self,
        track_id: str,
        frame: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
        prev_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    ):
        new_features = self.extract_appearance_features(frame, bbox_xyxy)
        if track_id not in self.track_features:
            self.track_features[track_id] = AppearanceFeatures()
        self.track_features[track_id].update(new_features)
        if prev_bbox_xyxy is not None:
            cx_new = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
            cy_new = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
            cx_old = (prev_bbox_xyxy[0] + prev_bbox_xyxy[2]) / 2
            cy_old = (prev_bbox_xyxy[1] + prev_bbox_xyxy[3]) / 2
            dx, dy = cx_new - cx_old, cy_new - cy_old
            if track_id not in self.velocity_history:
                self.velocity_history[track_id] = deque(maxlen=10)
            self.velocity_history[track_id].append((dx, dy))

    def get_smoothed_velocity(self, track_id: str) -> Tuple[float, float]:
        if track_id not in self.velocity_history or not self.velocity_history[track_id]:
            return (0.0, 0.0)
        velocities = list(self.velocity_history[track_id])
        weights = np.exp(np.linspace(-1, 0, len(velocities)))
        weights /= weights.sum()
        vx = sum(v[0] * w for v, w in zip(velocities, weights))
        vy = sum(v[1] * w for v, w in zip(velocities, weights))
        return (vx, vy)

    def add_lost_track(
        self,
        track_id: str,
        cls_id: Optional[int],
        device_id: Optional[str],
        bbox_xyxy: Tuple[int, int, int, int],
        bbox_norm: Tuple[float, float, float, float],
        frame_size: Tuple[int, int],
        first_seen: float,
        last_seen: float,
        total_age: int,
        total_hits: int,
        was_stable: bool,
    ):
        if total_hits < self.min_track_quality:
            return
        appearance = self.track_features.get(track_id, AppearanceFeatures())
        velocity = self.get_smoothed_velocity(track_id)
        self.lost_tracks[track_id] = LostTrack(
            track_id=track_id,
            cls_id=cls_id,
            device_id=device_id,
            last_bbox_xyxy=bbox_xyxy,
            last_bbox_norm=bbox_norm,
            velocity=velocity,
            appearance=appearance,
            lost_time=time.time(),
            first_seen=first_seen,
            last_seen=last_seen,
            total_age=total_age,
            total_hits=total_hits,
            was_stable=was_stable,
            frame_size=frame_size,
        )

    def remove_track(self, track_id: str):
        self.track_features.pop(track_id, None)
        self.velocity_history.pop(track_id, None)
        self.lost_tracks.pop(track_id, None)

    def cleanup_expired(self):
        current_time = time.time()
        for track_id in list(self.lost_tracks.keys()):
            if current_time - self.lost_tracks[track_id].lost_time > self.reid_buffer_time:
                self.lost_tracks.pop(track_id, None)
                self.track_features.pop(track_id, None)
                self.velocity_history.pop(track_id, None)

    def find_match(
        self,
        frame: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
        cls_id: Optional[int],
        device_id: Optional[str] = None,
    ) -> Optional[LostTrack]:
        self.cleanup_expired()
        if not self.lost_tracks:
            return None
        self.stats["total_reids"] += 1
        det_features = self.extract_appearance_features(frame, bbox_xyxy)
        frame_h, frame_w = frame.shape[:2]
        frame_size = (frame_w, frame_h)
        current_time = time.time()
        best_match = None
        best_score = 0.0
        for track_id, lost_track in self.lost_tracks.items():
            if device_id is not None and lost_track.device_id is not None and device_id != lost_track.device_id:
                continue
            if cls_id is not None and lost_track.cls_id is not None and cls_id != lost_track.cls_id:
                continue
            time_elapsed = current_time - lost_track.lost_time
            frames_elapsed = int(time_elapsed * 30)
            predicted_bbox = lost_track.get_predicted_position(frames_elapsed)
            iou_val = self.iou_xyxy(bbox_xyxy, predicted_bbox)
            if iou_val >= self.reid_iou_threshold:
                position_score = min(1.0, iou_val * 2)
            else:
                position_score = self.compute_position_similarity(bbox_xyxy, predicted_bbox, frame_size)
            appearance_score = self.compute_appearance_similarity(det_features, lost_track.appearance)
            size_score = self.compute_size_similarity(bbox_xyxy, lost_track.last_bbox_xyxy)
            total_score = (
                self.reid_position_weight * position_score
                + self.reid_appearance_weight * appearance_score
                + self.reid_size_weight * size_score
            )
            if lost_track.was_stable:
                total_score *= 1.1
            total_score += min(0.1, lost_track.total_hits / 100.0)
            total_score -= min(0.2, time_elapsed / self.reid_buffer_time * 0.2)
            if total_score > best_score:
                best_score = total_score
                best_match = lost_track
        if best_match is not None and best_score >= self.reid_appearance_threshold:
            self.stats["successful_reids"] += 1
            self.lost_tracks.pop(best_match.track_id, None)
            return best_match
        self.stats["failed_reids"] += 1
        return None

    def clear(self):
        self.lost_tracks.clear()
        self.track_features.clear()
        self.velocity_history.clear()


_reid_manager_instance: Optional[ReIDManager] = None


def get_reid_manager(
    reid_buffer_time: float = 30.0,
    reid_iou_threshold: float = 0.15,
    reid_appearance_threshold: float = 0.6,
    **kwargs,
) -> ReIDManager:
    global _reid_manager_instance
    if _reid_manager_instance is None:
        _reid_manager_instance = ReIDManager(
            reid_buffer_time=reid_buffer_time,
            reid_iou_threshold=reid_iou_threshold,
            reid_appearance_threshold=reid_appearance_threshold,
            **kwargs,
        )
    return _reid_manager_instance


def reset_reid_manager():
    global _reid_manager_instance
    if _reid_manager_instance is not None:
        _reid_manager_instance.clear()
    _reid_manager_instance = None
