from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from sklearn.cluster import KMeans

from sportvision.workflows._compat import (
    OBJECT_DETECTION_PREDICTION_KIND,
    BlockResult,
    OutputDefinition,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowBlock,
    WorkflowBlockManifest,
    WorkflowImageData,
    WorkflowImageSelector,
)

SHORT_DESCRIPTION = (
    "Classify detected players into teams using jersey color clustering."
)
LONG_DESCRIPTION = """
Uses HSV histogram features from the jersey region of each player crop,
then clusters with KMeans to assign team IDs. Maintains a fitted model
across frames for temporal consistency.
"""


class TeamClassifierManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Team Classifier",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["sportvision/team_classifier@v1"]
    image: WorkflowImageSelector | StepOutputImageSelector = Field(
        description="Input image for cropping player regions.",
    )
    detections: StepOutputSelector(
        kind=[OBJECT_DETECTION_PREDICTION_KIND],
    ) = Field(
        description="Player detections with bounding boxes.",
    )
    n_teams: int = Field(
        default=2,
        ge=2,
        le=10,
        description="Number of teams to cluster into.",
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


class TeamClassifierBlockV1(WorkflowBlock):
    def __init__(self):
        self._model: KMeans | None = None
        self._n_teams: int = 2

    @classmethod
    def get_manifest(cls) -> type[WorkflowBlockManifest]:
        return TeamClassifierManifest

    @staticmethod
    def _extract_features(crop: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        jersey = hsv[int(h * 0.2) : int(h * 0.6), int(w * 0.2) : int(w * 0.8)]
        if jersey.size == 0:
            jersey = hsv
        hist_h = cv2.calcHist([jersey], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([jersey], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([jersey], [2], None, [8], [0, 256]).flatten()
        feat = np.concatenate([hist_h, hist_s, hist_v])
        feat = feat / (feat.sum() + 1e-8)
        return feat

    def _get_crops(
        self, image: np.ndarray, detections: sv.Detections
    ) -> list[np.ndarray]:
        crops = []
        for xyxy in detections.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((10, 10, 3), dtype=np.uint8)
            crops.append(crop)
        return crops

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        n_teams: int = 2,
    ) -> BlockResult:
        if len(detections) == 0:
            detections.data["team_id"] = np.array([], dtype=int)
            return {"detections": detections}

        img = image.numpy_image
        crops = self._get_crops(img, detections)
        features = np.array([self._extract_features(c) for c in crops])

        if self._model is None or self._n_teams != n_teams:
            self._n_teams = n_teams
            self._model = KMeans(n_clusters=n_teams, random_state=0, n_init=10)
            self._model.fit(features)

        team_ids = self._model.predict(features)
        detections.data["team_id"] = team_ids
        return {"detections": detections}
