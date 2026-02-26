from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sportvision.analytics.possession import PossessionTracker
from sportvision.annotators import StatsOverlayAnnotator, TeamColorAnnotator
from sportvision.detection import SportsDetector
from sportvision.tracking import SportsTracker


@dataclass
class SportVisionPipeline:
    sport: str = "soccer"
    device: str = "cpu"
    _detector: SportsDetector = field(default=None, init=False)
    _tracker: SportsTracker = field(default=None, init=False)
    _possession: PossessionTracker = field(default=None, init=False)
    _team_annotator: TeamColorAnnotator = field(default=None, init=False)
    _stats_annotator: StatsOverlayAnnotator = field(default=None, init=False)

    def __post_init__(self):
        self._detector = SportsDetector()
        self._tracker = SportsTracker()
        self._possession = PossessionTracker()
        self._team_annotator = TeamColorAnnotator()
        self._stats_annotator = StatsOverlayAnnotator()

    def process_frame(self, frame: np.ndarray) -> dict:
        detections = self._detector.detect(frame)
        tracked = self._tracker.update(detections)
        annotated = self._team_annotator.annotate(
            frame.copy(),
            tracked["xyxy"],
            tracked.get("class_id", np.zeros(len(tracked["xyxy"]))),
        )
        stats = self._possession.get_stats()
        annotated = self._stats_annotator.annotate(annotated, possession=stats)
        return {"detections": tracked, "annotated_frame": annotated}

    def get_stats(self) -> dict:
        return {"possession": self._possession.get_stats()}
