from __future__ import annotations

from typing import Literal

import cv2
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

SHORT_DESCRIPTION = "Calculate cumulative pixel distance traveled per tracked object."
LONG_DESCRIPTION = """
Stores the center position of each tracked detection across frames and
computes the cumulative path distance. Requires detections to have
tracker_id assigned.
"""


class DistanceCalculatorManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Distance Calculator",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
        }
    )
    type: Literal["sportvision/distance_calculator@v1"]
    detections: StepOutputSelector(
        kind=[OBJECT_DETECTION_PREDICTION_KIND],
    ) = Field(
        description="Detections with tracker_id assigned.",
    )
    homography_matrix: list[list[float]] | None = Field(
        default=None,
        description="Optional 3x3 homography matrix to convert pixel "
        "coordinates to field coordinates. When provided, "
        "distances are in field units (e.g. meters).",
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


class DistanceCalculatorBlockV1(WorkflowBlock):
    def __init__(self):
        self._positions: dict[int, list[tuple[float, float]]] = {}
        self._cumulative: dict[int, float] = {}

    @classmethod
    def get_manifest(cls) -> type[WorkflowBlockManifest]:
        return DistanceCalculatorManifest

    def run(
        self,
        detections: sv.Detections,
        homography_matrix: list[list[float]] | None = None,
    ) -> BlockResult:
        if len(detections) == 0:
            detections.data["distance"] = np.array([], dtype=float)
            return {"detections": detections}

        tracker_ids = detections.tracker_id
        if tracker_ids is None:
            detections.data["distance"] = np.zeros(len(detections))
            return {"detections": detections}

        centers_x = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
        centers_y = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2

        if homography_matrix is not None:
            h_mat = np.array(homography_matrix, dtype=np.float64)
            pts = np.column_stack([centers_x, centers_y]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(pts, h_mat).reshape(-1, 2)
            centers_x = transformed[:, 0]
            centers_y = transformed[:, 1]

        distances = np.zeros(len(detections))
        for i, tid in enumerate(tracker_ids):
            tid = int(tid)
            cx, cy = float(centers_x[i]), float(centers_y[i])

            if tid in self._positions:
                prev = self._positions[tid][-1]
                step = np.sqrt((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2)
                self._cumulative[tid] = self._cumulative.get(tid, 0.0) + step
                self._positions[tid].append((cx, cy))
            else:
                self._positions[tid] = [(cx, cy)]
                self._cumulative[tid] = 0.0

            distances[i] = self._cumulative[tid]

        detections.data["distance"] = distances
        return {"detections": detections}
