from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeedEstimator:
    fps: float = 30.0

    def calculate(
        self,
        positions: dict[int, list[tuple[float, float]]],
        frame_indices: dict[int, list[int]],
    ) -> dict[int, float]:
        speeds = {}
        for tid, pos_list in positions.items():
            if len(pos_list) < 2:
                speeds[tid] = 0.0
                continue
            frames = frame_indices[tid]
            p1, p2 = np.array(pos_list[-2]), np.array(pos_list[-1])
            dt = (frames[-1] - frames[-2]) / self.fps
            if dt <= 0:
                speeds[tid] = 0.0
                continue
            dist = float(np.linalg.norm(p2 - p1))
            speeds[tid] = (dist / dt) * 3.6  # m/s -> km/h
        return speeds
