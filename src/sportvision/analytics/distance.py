from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DistanceCalculator:
    def calculate(
        self, positions: dict[int, list[tuple[float, float]]]
    ) -> dict[int, float]:
        distances = {}
        for tid, pos_list in positions.items():
            if len(pos_list) < 2:
                distances[tid] = 0.0
                continue
            pts = np.array(pos_list)
            diffs = np.diff(pts, axis=0)
            distances[tid] = float(np.sum(np.linalg.norm(diffs, axis=1)))
        return distances
