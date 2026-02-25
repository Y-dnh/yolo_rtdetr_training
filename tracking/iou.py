"""
IoU для bbox у форматі (cx, cy, w, h) нормалізованих координат.
"""


def iou(boxA, boxB):
    """
    Обчислення IoU між двома боксами.
    Формат боксів: (cx, cy, w, h). Внутрішньо перетворюється на (x1, y1, x2, y2).
    """
    def to_corners(box):
        cx, cy, w, h = box
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    A = to_corners(boxA)
    B = to_corners(boxB)
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)
