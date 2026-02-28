from __future__ import annotations

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

SHORT_DESCRIPTION = "Filter and remap COCO detections to sports-specific classes."
LONG_DESCRIPTION = """
Filters detections to only keep sports-relevant classes (e.g. person, sports ball)
and remaps their class IDs to a sports-specific scheme (player=0, ball=1).
"""

# COCO class IDs: person=0, sports_ball=32
DEFAULT_CLASS_MAPPING: dict[int, int] = {
    0: 0,  # person -> player
    32: 1,  # sports_ball -> ball
}


class SportsDetectionFilterManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Sports Detection Filter",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["sportvision/sports_detection_filter@v1"]
    detections: StepOutputSelector(
        kind=[OBJECT_DETECTION_PREDICTION_KIND],
    ) = Field(
        description="Input detections from a COCO-trained model.",
    )
    class_mapping: dict[int, int] = Field(
        default=DEFAULT_CLASS_MAPPING,
        description="Mapping from source class_id to target class_id.",
    )

    @classmethod
    def describe_outputs(cls) -> list[OutputDefinition]:
        return [
            OutputDefinition(
                name="detections",
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> str | None:
        return ">=1.0.0,<2.0.0"


class SportsDetectionFilterBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> type[WorkflowBlockManifest]:
        return SportsDetectionFilterManifest

    def run(
        self,
        detections: sv.Detections,
        class_mapping: dict[int, int] | None = None,
    ) -> BlockResult:
        if class_mapping is None:
            class_mapping = DEFAULT_CLASS_MAPPING

        if len(detections) == 0:
            return {"detections": detections}

        allowed = set(class_mapping.keys())
        mask = np.array([cid in allowed for cid in detections.class_id])
        filtered = detections[mask]

        if len(filtered) > 0:
            filtered.class_id = np.array(
                [class_mapping[cid] for cid in filtered.class_id]
            )

        return {"detections": filtered}
