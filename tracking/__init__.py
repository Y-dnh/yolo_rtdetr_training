"""
Модуль трекінгу: детекція кожні N фреймів + NanoTrack між ними.
Конфіг (шляхи, інтервал, класи) — у track.py.
"""

from tracking.nano_tracker import NanoTracker, TrackedObject
from tracking.iou import iou
from tracking.reid_manager import get_reid_manager, ReIDManager, LostTrack

__all__ = [
    "NanoTracker",
    "TrackedObject",
    "get_reid_manager",
    "ReIDManager",
    "LostTrack",
    "iou",
]
