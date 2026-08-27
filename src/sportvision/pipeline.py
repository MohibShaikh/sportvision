from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sportvision.analytics.possession import PossessionTracker
from sportvision.annotators import StatsOverlayAnnotator, TeamColorAnnotator
from sportvision.detection import SportsDetector
from sportvision.teams import TeamClassifier
from sportvision.tracking import SportsTracker


@dataclass
class SportVisionPipeline:
    device: str = "auto"
    n_teams: int = 2
    detect_every: int = 1
    infer_size: int | None = 640
    _detector: SportsDetector = field(default=None, init=False)
    _tracker: SportsTracker = field(default=None, init=False)
    _possession: PossessionTracker = field(default=None, init=False)
    _team_classifier: TeamClassifier = field(default=None, init=False)
    _team_annotator: TeamColorAnnotator = field(default=None, init=False)
    _stats_annotator: StatsOverlayAnnotator = field(default=None, init=False)
    _tracked: dict[str, np.ndarray] | None = field(default=None, init=False)
    _frame_idx: int = field(default=0, init=False)

    def __post_init__(self):
        self._detector = SportsDetector(device=self.device, infer_size=self.infer_size)
        self._tracker = SportsTracker()
        self._possession = PossessionTracker()
        self._team_classifier = TeamClassifier(n_teams=self.n_teams)
        self._team_annotator = TeamColorAnnotator()
        self._stats_annotator = StatsOverlayAnnotator()

    def process_frame(self, frame: np.ndarray) -> dict:
        if self._frame_idx % self.detect_every == 0:
            detections = self._detector.detect(frame)
            self._tracked = self._tracker.update(detections)
        tracked = self._tracked

        player_mask = tracked["class_id"] == 0
        crops = self._crops(frame, tracked["xyxy"][player_mask])
        if self._team_classifier._model is None:
            if len(crops) >= 2:
                self._team_classifier.fit(crops)
            team_ids = np.zeros(tracked["xyxy"].shape[0], dtype=int)
        else:
            predicted = self._team_classifier.predict(crops)
            team_ids = -np.ones(tracked["xyxy"].shape[0], dtype=int)
            team_ids[player_mask] = predicted

        self._update_possession(tracked, team_ids, player_mask)

        annotated = self._team_annotator.annotate(
            frame.copy(), tracked["xyxy"], team_ids
        )
        stats = self._possession.get_stats()
        annotated = self._stats_annotator.annotate(annotated, possession=stats)
        self._frame_idx += 1
        return {"detections": tracked, "annotated_frame": annotated}

    def _update_possession(
        self,
        tracked: dict[str, np.ndarray],
        team_ids: np.ndarray,
        player_mask: np.ndarray,
    ) -> None:
        if self._team_classifier._model is None:
            return
        centers = (tracked["xyxy"][:, :2] + tracked["xyxy"][:, 2:]) / 2
        ball = centers[~player_mask]
        players = centers[player_mask]
        if len(players) == 0 or len(ball) == 0:
            return
        self._possession.update(players, team_ids[player_mask], ball[0])

    @staticmethod
    def _crops(frame: np.ndarray, xyxy: np.ndarray) -> list[np.ndarray]:
        h, w = frame.shape[:2]
        crops = []
        for x1, y1, x2, y2 in xyxy.astype(int):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                crops.append(np.zeros((20, 20, 3), dtype=np.uint8))
            else:
                crops.append(crop)
        return crops

    def get_stats(self) -> dict:
        return {"possession": self._possession.get_stats()}
