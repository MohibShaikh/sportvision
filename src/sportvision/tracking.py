from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SportsTracker:
    tracker_type: str = "bytetrack"
    track_thresh: float = 0.25
    match_thresh: float = 0.1
    _tracker: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.tracker_type != "bytetrack":
            raise ValueError("Only bytetrack is supported")
        for value in (self.track_thresh, self.match_thresh):
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Tracking thresholds must be in (0, 1]")

    def _init_tracker(self) -> None:
        try:
            from trackers import ByteTrackTracker

            self._tracker = ByteTrackTracker(
                track_activation_threshold=self.track_thresh,
                minimum_iou_threshold=self.match_thresh,
            )
        except ImportError as exc:
            raise RuntimeError("Tracking requires the 'trackers' package") from exc

    def update(self, detections: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._tracker is None:
            self._init_tracker()

        # Ball observations are evidence even before a persistent track exists.
        # Keep them out of person association to prevent cross-class ID matches.
        classes = detections["class_id"]
        people = classes != 1
        balls = ~people
        xyxy = detections["xyxy"][people]
        import supervision as sv

        sv_dets = sv.Detections(
            xyxy=xyxy,
            class_id=classes[people],
            confidence=detections["confidence"][people],
        )
        tracked = self._tracker.update(sv_dets)
        return {
            "xyxy": np.concatenate([tracked.xyxy, detections["xyxy"][balls]]),
            "class_id": np.concatenate([tracked.class_id, classes[balls]]),
            "confidence": np.concatenate(
                [tracked.confidence, detections["confidence"][balls]]
            ),
            "tracker_id": np.concatenate(
                [tracked.tracker_id, np.full(balls.sum(), -1, dtype=int)]
            ),
        }

    def reset(self) -> None:
        self._tracker = None
