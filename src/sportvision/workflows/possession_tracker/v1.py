from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from sportvision.workflows._compat import (
    OBJECT_DETECTION_PREDICTION_KIND,
    BlockResult,
    OutputDefinition,
    StepOutputSelector,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Track ball possession across teams over time."
LONG_DESCRIPTION = """
Identifies the ball detection (class_id=1), finds the nearest player,
and accumulates possession counts per team. Requires detections to have
tracker_id and team_id in their data dict.
"""


class PossessionTrackerManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Possession Tracker",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
        }
    )
    type: Literal["sportvision/possession_tracker@v1"]
    detections: StepOutputSelector(
        kind=[OBJECT_DETECTION_PREDICTION_KIND],
    ) = Field(
        description="Detections with tracker_id, class_id, and team_id.",
    )
    ball_class_id: int = Field(
        default=1,
        description="Class ID that represents the ball.",
    )
    ball_proximity_threshold: float = Field(
        default=50.0,
        ge=0.0,
        description="Max pixel distance for possession attribution.",
    )

    @classmethod
    def describe_outputs(cls) -> list[OutputDefinition]:
        return [
            OutputDefinition(name="possession_stats"),
            OutputDefinition(name="possessing_team"),
            OutputDefinition(name="warning"),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> str | None:
        return ">=1.0.0,<2.0.0"


logger = logging.getLogger(__name__)


class PossessionTrackerBlockV1(WorkflowBlock):
    def __init__(self):
        self._counts: dict[int, int] = {}

    @classmethod
    def get_manifest(cls) -> type[WorkflowBlockManifest]:
        return PossessionTrackerManifest

    def run(
        self,
        detections: sv.Detections,
        ball_class_id: int = 1,
        ball_proximity_threshold: float = 50.0,
    ) -> BlockResult:
        if len(detections) == 0:
            return {
                "possession_stats": {},
                "possessing_team": -1,
                "warning": "",
            }

        class_ids = detections.class_id
        ball_mask = class_ids == ball_class_id
        player_mask = ~ball_mask

        if not ball_mask.any() or not player_mask.any():
            total = sum(self._counts.values())
            stats = {k: v / total for k, v in self._counts.items()} if total > 0 else {}
            possessing = max(self._counts, key=self._counts.get) if stats else -1
            return {
                "possession_stats": stats,
                "possessing_team": possessing,
                "warning": "",
            }

        # Ball center (use first ball detection)
        ball_idx = np.where(ball_mask)[0][0]
        ball_xyxy = detections.xyxy[ball_idx]
        ball_center = np.array(
            [(ball_xyxy[0] + ball_xyxy[2]) / 2, (ball_xyxy[1] + ball_xyxy[3]) / 2]
        )

        # Player centers
        player_indices = np.where(player_mask)[0]
        player_xyxy = detections.xyxy[player_indices]
        player_centers = np.column_stack(
            [
                (player_xyxy[:, 0] + player_xyxy[:, 2]) / 2,
                (player_xyxy[:, 1] + player_xyxy[:, 3]) / 2,
            ]
        )

        distances = np.linalg.norm(player_centers - ball_center, axis=1)
        nearest_local = int(np.argmin(distances))

        warning = ""
        if distances[nearest_local] <= ball_proximity_threshold:
            nearest_global = player_indices[nearest_local]
            team_ids = detections.data.get("team_id")
            if team_ids is not None:
                team = int(team_ids[nearest_global])
                self._counts[team] = self._counts.get(team, 0) + 1
            else:
                warning = "team_id missing from detections; possession not updated"
                logger.warning(warning)

        total = sum(self._counts.values())
        stats = {k: v / total for k, v in self._counts.items()} if total > 0 else {}
        possessing = max(self._counts, key=self._counts.get) if stats else -1

        return {
            "possession_stats": stats,
            "possessing_team": possessing,
            "warning": warning,
        }
