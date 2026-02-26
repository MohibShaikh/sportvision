from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TeamColorAnnotator:
    home_color: tuple[int, int, int] = (255, 0, 0)
    away_color: tuple[int, int, int] = (0, 0, 255)
    thickness: int = 2

    def annotate(
        self, frame: np.ndarray, xyxy: np.ndarray, team_ids: np.ndarray
    ) -> np.ndarray:
        colors = {0: self.home_color, 1: self.away_color}
        for box, tid in zip(xyxy, team_ids):
            color = colors.get(int(tid), (128, 128, 128))
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
        return frame


@dataclass
class StatsOverlayAnnotator:
    position: str = "top-left"
    font_scale: float = 0.8

    def annotate(
        self,
        frame: np.ndarray,
        possession: dict[int, float] | None = None,
        **kwargs,
    ) -> np.ndarray:
        y = 30
        if possession:
            for team, pct in possession.items():
                text = f"Team {team}: {pct:.0%}"
                cv2.putText(
                    frame,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (255, 255, 255),
                    2,
                )
                y += 30
        return frame


@dataclass
class TrailAnnotator:
    trail_length: int = 30
    fade: bool = True
    thickness: int = 2

    def annotate(
        self, frame: np.ndarray, tracks: dict[int, list[tuple[float, float]]]
    ) -> np.ndarray:
        for _tid, positions in tracks.items():
            pts = positions[-self.trail_length :]
            for i in range(1, len(pts)):
                alpha = i / len(pts) if self.fade else 1.0
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                p2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(frame, p1, p2, color, self.thickness)
        return frame
