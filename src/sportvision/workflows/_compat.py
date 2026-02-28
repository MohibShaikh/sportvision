"""Compatibility shim for inference workflow types.

When the ``inference`` package is installed, re-exports its real types.
Otherwise, provides lightweight stand-ins so that sportvision blocks can
still be imported, instantiated, and tested without the full inference
dependency.
"""

from __future__ import annotations

from typing import Any

try:
    from inference.core.workflows.execution_engine.entities.base import (  # noqa: F401
        OutputDefinition,
        WorkflowImageData,
    )
    from inference.core.workflows.execution_engine.entities.types import (  # noqa: F401
        BATCH_OF_IMAGES_KIND,
        OBJECT_DETECTION_PREDICTION_KIND,
        StepOutputImageSelector,
        StepOutputSelector,
        WorkflowImageSelector,
    )
    from inference.core.workflows.prototypes.block import (  # noqa: F401
        BlockResult,
        WorkflowBlock,
        WorkflowBlockManifest,
    )

    HAS_INFERENCE = True

except ImportError:
    HAS_INFERENCE = False

    # -- Minimal stand-ins for type annotations and base classes ----------

    class OutputDefinition:  # type: ignore[no-redef]
        def __init__(self, name: str = "", kind: Any = None):
            self.name = name
            self.kind = kind

    class WorkflowImageData:  # type: ignore[no-redef]
        """Stand-in; real images are passed as objects with .numpy_image."""

        numpy_image: Any = None

    # Sentinel kind values
    BATCH_OF_IMAGES_KIND: Any = "batch_of_images"
    OBJECT_DETECTION_PREDICTION_KIND: Any = "object_detection_prediction"

    # Stand-in selector types — must support | operator and callable syntax
    # for use in Pydantic model annotations.
    StepOutputImageSelector = Any
    StepOutputSelector = lambda *_a, **_kw: Any  # noqa: E731
    WorkflowImageSelector = Any

    BlockResult = dict  # type: ignore[misc]

    from pydantic import BaseModel

    class WorkflowBlockManifest(BaseModel):  # type: ignore[no-redef]
        @classmethod
        def describe_outputs(cls) -> list[OutputDefinition]:
            return []

        @classmethod
        def get_execution_engine_compatibility(cls) -> str | None:
            return None

    class WorkflowBlock:  # type: ignore[no-redef]
        @classmethod
        def get_manifest(cls) -> type:
            raise NotImplementedError

        def run(self, **kwargs: Any) -> dict:
            raise NotImplementedError
