"""SportVision: Sports clip review and analytics building blocks."""

__version__ = "0.4.0"

from sportvision.pipeline import SportVisionPipeline

__all__ = ["SportVisionPipeline"]


def load_blocks():
    """Support WORKFLOWS_PLUGINS=sportvision as well as sportvision.workflows."""
    from sportvision.workflows import load_blocks as workflow_blocks

    return workflow_blocks()


def load_kinds():
    from sportvision.workflows import load_kinds as workflow_kinds

    return workflow_kinds()
