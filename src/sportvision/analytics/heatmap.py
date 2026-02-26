from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class HeatmapGenerator:
    resolution: tuple[int, int] = (105, 68)  # (width, height)
    _grids: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def _get_grid(self, team: int) -> np.ndarray:
        if team not in self._grids:
            w, h = self.resolution
            self._grids[team] = np.zeros((h, w), dtype=np.float64)
        return self._grids[team]

    def update(self, positions: np.ndarray, team_ids: np.ndarray) -> None:
        w, h = self.resolution
        for pos, tid in zip(positions, team_ids):
            tid = int(tid)
            grid = self._get_grid(tid)
            x, y = int(np.clip(pos[0], 0, w - 1)), int(np.clip(pos[1], 0, h - 1))
            grid[y, x] += 1.0

    def render(self, team: int = 0) -> np.ndarray:
        w, h = self.resolution
        grid = self._get_grid(team)
        if grid.max() > 0:
            normalized = (grid / grid.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros((h, w), dtype=np.uint8)
        blurred = cv2.GaussianBlur(normalized, (0, 0), sigmaX=2, sigmaY=2)
        colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
        mask = blurred == 0
        colored[mask] = 0
        return colored

    def reset(self) -> None:
        self._grids.clear()
