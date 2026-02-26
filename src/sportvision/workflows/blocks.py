from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkflowBlock:
    _type: str = ""

    def get_manifest(self) -> dict:
        return {"type": self._type}


@dataclass
class TeamClassifierBlock(WorkflowBlock):
    _type: str = "sportvision/team_classifier"

    def run(self, image, detections) -> dict:
        from sportvision.teams import TeamClassifier

        _ = TeamClassifier
        return {"predictions": detections, "team_ids": []}


@dataclass
class PossessionTrackerBlock(WorkflowBlock):
    _type: str = "sportvision/possession_tracker"

    def run(self, detections) -> dict:
        from sportvision.analytics.possession import PossessionTracker

        tracker = PossessionTracker()
        return {"stats": tracker.get_stats()}


@dataclass
class SpeedEstimatorBlock(WorkflowBlock):
    _type: str = "sportvision/speed_estimator"


@dataclass
class HeatmapGeneratorBlock(WorkflowBlock):
    _type: str = "sportvision/heatmap_generator"


@dataclass
class StatsOverlayBlock(WorkflowBlock):
    _type: str = "sportvision/stats_overlay"
