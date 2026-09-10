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
    refit_every: int = Field(
        default=0,
        ge=0,
        description="Refit KMeans every N frames. 0 means fit once and reuse.",
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
        self._frame_count: int = 0

    @classmethod
    def get_manifest(cls) -> type[WorkflowBlockManifest]:
        return TeamClassifierManifest

    @staticmethod
    def _extract_features(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        jersey = crop[int(h * 0.2) : int(h * 0.6), int(w * 0.2) : int(w * 0.8)]
        if jersey.size == 0:
            jersey = crop
        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
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
        refit_every: int = 0,
    ) -> BlockResult:
        if len(detections) == 0:
            detections.data["team_id"] = np.array([], dtype=int)
            return {"detections": detections}

        self._frame_count += 1
        img = image.numpy_image
        if n_teams < 1 or refit_every < 0:
            raise ValueError("n_teams must be positive and refit_every nonnegative")
        player_mask = (
            detections.class_id == 0
            if detections.class_id is not None
            else np.ones(len(detections), dtype=bool)
        )
        team_ids = np.full(len(detections), -1, dtype=int)
        detections.data["team_id"] = team_ids
        if not player_mask.any():
            return {"detections": detections}
        crops = self._get_crops(img, detections[player_mask])
        features = np.array([self._extract_features(c) for c in crops])

        needs_refit = (
            refit_every > 0
            and self._model is not None
            and self._frame_count % refit_every == 0
        )

        if self._model is None or self._n_teams != n_teams or needs_refit:
            if len(features) < n_teams:
                return {"detections": detections}
            previous = self._model if self._n_teams == n_teams else None
            self._n_teams = n_teams
            self._model = KMeans(
                n_clusters=n_teams,
                random_state=0,
                init=previous.cluster_centers_ if previous is not None else "k-means++",
                n_init=1 if previous is not None else 2,
            )
            self._model.fit(features)

        team_ids[player_mask] = self._model.predict(features)
        detections.data["team_id"] = team_ids
        return {"detections": detections}
