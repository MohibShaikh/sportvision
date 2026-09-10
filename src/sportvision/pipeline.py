from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sportvision.analytics.possession import PossessionTracker
from sportvision.annotators import StatsOverlayAnnotator, TeamColorAnnotator
from sportvision.detection import SportsDetector
from sportvision.review import possession_evidence
from sportvision.teams import TeamClassifier
from sportvision.tracking import SportsTracker


@dataclass
class SportVisionPipeline:
    model: str = "rfdetr-base"
    device: str = "auto"
    n_teams: int = 2
    detect_every: int = 1
    infer_size: int | None = 640
    possession_method: str = "center"
    confidence_threshold: float = 0.25
    ball_confidence_threshold: float | None = None
    _detector: SportsDetector = field(default=None, init=False)
    _tracker: SportsTracker = field(default=None, init=False)
    _possession: PossessionTracker = field(default=None, init=False)
    _team_classifier: TeamClassifier = field(default=None, init=False)
    _team_annotator: TeamColorAnnotator = field(default=None, init=False)
    _stats_annotator: StatsOverlayAnnotator = field(default=None, init=False)
    _tracked: dict[str, np.ndarray] | None = field(default=None, init=False)
    _frame_idx: int = field(default=0, init=False)

    def __post_init__(self):
        if self.possession_method not in ("center", "body"):
            raise ValueError("Possession method must be center or body")
        if self.detect_every < 1 or self.n_teams < 1:
            raise ValueError("detect_every and n_teams must be positive")
        self._detector = SportsDetector(
            model=self.model,
            device=self.device,
            infer_size=self.infer_size,
            confidence_threshold=self.confidence_threshold,
            ball_confidence_threshold=self.ball_confidence_threshold,
        )
        self._tracker = SportsTracker()
        self._possession = PossessionTracker(n_teams=self.n_teams)
        self._team_classifier = TeamClassifier(n_teams=self.n_teams)
        self._team_annotator = TeamColorAnnotator()
        self._stats_annotator = StatsOverlayAnnotator()

    def set_team_colors(self, colors: list[tuple[int, int, int]]) -> None:
        """Initialize teams from BGR jersey colors before processing a new clip."""
        if self._frame_idx:
            raise ValueError(
                "Choose teams before processing; start a new pipeline to change them"
            )
        samples = [np.full((40, 40, 3), color, np.uint8) for color in colors]
        self._team_classifier.set_team_samples(samples)

    def process_frame(self, frame: np.ndarray) -> dict:
        fresh = self._frame_idx % self.detect_every == 0
        if fresh:
            detections = self._detector.detect(frame)
            tracked = self._tracker.update(detections)
            player_mask = tracked["class_id"] == 0
            crops = self._crops(frame, tracked["xyxy"][player_mask])
            if not self._team_classifier.is_fitted and len(crops) >= self.n_teams:
                self._team_classifier.fit(crops)
            team_ids = np.full(len(tracked["xyxy"]), -1, dtype=int)
            if self._team_classifier.is_fitted:
                team_ids[player_mask] = self._team_classifier.predict(crops)
            tracked["team_id"] = team_ids
            self._update_possession(tracked, team_ids, player_mask)
            self._tracked = tracked
        tracked = self._tracked
        team_ids = tracked["team_id"]

        annotated = self._team_annotator.annotate(
            frame.copy(), tracked["xyxy"], team_ids
        )
        stats = self._possession.get_stats()
        annotated = self._stats_annotator.annotate(annotated, possession=stats)
        self._frame_idx += 1
        return {
            "detections": tracked,
            "annotated_frame": annotated,
            "fresh_detections": fresh,
            "stats": self.get_stats(),
        }

    def _update_possession(
        self,
        tracked: dict[str, np.ndarray],
        team_ids: np.ndarray,
        player_mask: np.ndarray,
    ) -> None:
        evidence = possession_evidence(
            {**tracked, "team_id": team_ids},
            self._possession.ball_proximity_thresh,
            method=self.possession_method,
        )
        if evidence["reason"] == "attributed":
            team = evidence["team_id"]
            self._possession._counts[team] = self._possession._counts.get(team, 0) + 1

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
