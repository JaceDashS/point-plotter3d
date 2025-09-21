import numpy as np
from typing import List, Tuple

class PointsModel:
    """단순 포인트 저장/조회 모델"""
    def __init__(self):
        self._points: List[Tuple[float, float, float]] = []

    def add(self, x: float, y: float, z: float):
        self._points.append((float(x), float(y), float(z)))

    def clear(self):
        self._points.clear()

    def as_numpy(self) -> np.ndarray:
        if not self._points:
            return np.zeros((0, 3), dtype=float)
        return np.array(self._points, dtype=float)

    def __len__(self):
        return len(self._points)
