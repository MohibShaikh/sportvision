from __future__ import annotations

from sportvision.workflows.distance_calculator.v1 import DistanceCalculatorBlockV1
from sportvision.workflows.possession_tracker.v1 import PossessionTrackerBlockV1
from sportvision.workflows.sports_detection_filter.v1 import (
    SportsDetectionFilterBlockV1,
)
from sportvision.workflows.team_classifier.v1 import TeamClassifierBlockV1


def load_blocks() -> list[type]:
    return [
        TeamClassifierBlockV1,
        PossessionTrackerBlockV1,
        DistanceCalculatorBlockV1,
        SportsDetectionFilterBlockV1,
    ]


def load_kinds():
    from sportvision.workflows.kinds import SPORTS_DETECTION_KIND

    return [SPORTS_DETECTION_KIND]


__all__ = [
    "TeamClassifierBlockV1",
    "PossessionTrackerBlockV1",
    "DistanceCalculatorBlockV1",
    "SportsDetectionFilterBlockV1",
    "load_blocks",
    "load_kinds",
]
