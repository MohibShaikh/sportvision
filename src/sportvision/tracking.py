from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SportsTracker:
    tracker_type: str = "bytetrack"
    track_thresh: float = 0.25
    match_thresh: float = 0.8
    _tracker: Any = field(default=None, init=False, repr=False)
    _next_id: int = field(default=0, init=False, repr=False)

    def _init_tracker(self) -> None:
        try:
            from trackers import ByteTrack

            self._tracker = ByteTrack(
                track_activation_threshold=self.track_thresh,
                minimum_matching_threshold=self.match_thresh,
            )
        except ImportError:
            self._tracker = None

    def update(self, detections: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._tracker is None:
            self._init_tracker()

        xyxy = detections["xyxy"]
        n = len(xyxy)

        if self._tracker is not None:
            try:
                import supervision as sv

                sv_dets = sv.Detections(
                    xyxy=xyxy,
                    class_id=detections.get("class_id"),
                    confidence=detections.get("confidence"),
                )
                tracked = self._tracker.update_with_detections(sv_dets)
                return {
                    "xyxy": tracked.xyxy,
                    "class_id": tracked.class_id,
                    "confidence": tracked.confidence,
                    "tracker_id": tracked.tracker_id,
                }
            except Exception:
                pass

        # Fallback: assign sequential IDs
        ids = np.arange(self._next_id, self._next_id + n)
        self._next_id += n
        return {**detections, "tracker_id": ids}

    def reset(self) -> None:
        self._tracker = None
        self._next_id = 0
