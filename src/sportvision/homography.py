from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FieldHomography:
    field_length: float = 105
    field_width: float = 68
    _matrix: np.ndarray | None = field(default=None, init=False, repr=False)

    @classmethod
    def soccer(
        cls, field_length: float = 105, field_width: float = 68
    ) -> FieldHomography:
        return cls(field_length=field_length, field_width=field_width)

    @classmethod
    def basketball(cls) -> FieldHomography:
        return cls(field_length=28.65, field_width=15.24)

    @classmethod
    def tennis(cls) -> FieldHomography:
        return cls(field_length=23.77, field_width=10.97)

    def set_keypoints(self, src_points: np.ndarray, dst_points: np.ndarray) -> None:
        if len(src_points) < 4 or len(dst_points) < 4:
            raise ValueError("At least 4 keypoint pairs required")
        self._matrix, _ = cv2.findHomography(src_points, dst_points)

    def transform(self, points: np.ndarray) -> np.ndarray:
        if self._matrix is None:
            raise RuntimeError("Must call set_keypoints() first")
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, self._matrix)
        return transformed.reshape(-1, 2)
