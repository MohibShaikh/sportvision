# SportVision

Real-time sports analytics toolkit built on the Roboflow ecosystem. Detects players and ball via RF-DETR, tracks with ByteTrack, classifies teams by jersey color, and computes possession, speed, distance, and heatmaps.

## Features

- **Detection** — RF-DETR (base/large) with COCO→sports class mapping
- **Tracking** — ByteTrack via supervision with fallback sequential IDs
- **Team Classification** — KMeans on HSV jersey histograms
- **Homography** — Pixel→field coordinate mapping via `cv2.findHomography`
- **Analytics** — Possession, speed, distance, and heatmap generation
- **Annotation** — Team-colored bounding boxes, stats overlay, player trails

## Quick Start

```bash
pip install sportvision
```

```python
from sportvision.pipeline import SportVisionPipeline

pipeline = SportVisionPipeline()
result = pipeline.process_frame(frame)
```

## Roboflow Workflows Plugin

SportVision ships as a [Roboflow Workflows](https://inference.roboflow.com/workflows/about/) plugin. Install with inference and activate:

```bash
pip install "sportvision[workflows]"
export WORKFLOWS_PLUGINS="sportvision.workflows"
```

This registers 4 blocks you can use in any Roboflow Workflow:

| Block | Type Identifier | Description |
|-------|----------------|-------------|
| Team Classifier | `sportvision/team_classifier@v1` | Clusters players into teams by jersey color |
| Possession Tracker | `sportvision/possession_tracker@v1` | Tracks ball possession per team over time |
| Distance Calculator | `sportvision/distance_calculator@v1` | Cumulative pixel distance per tracked player |
| Sports Detection Filter | `sportvision/sports_detection_filter@v1` | Filters COCO detections to sports classes |

### Example: Using blocks directly in Python

```python
import cv2
import numpy as np
import supervision as sv

from sportvision.workflows.team_classifier.v1 import TeamClassifierBlockV1
from sportvision.workflows.possession_tracker.v1 import PossessionTrackerBlockV1
from sportvision.workflows.distance_calculator.v1 import DistanceCalculatorBlockV1
from sportvision.workflows.sports_detection_filter.v1 import SportsDetectionFilterBlockV1

# --- Filter COCO detections to sports classes ---
det_filter = SportsDetectionFilterBlockV1()
# Assume `raw_detections` comes from a COCO model (person=0, sports_ball=32)
result = det_filter.run(detections=raw_detections)
detections = result["detections"]  # now player=0, ball=1

# --- Classify players into teams ---
team_block = TeamClassifierBlockV1()
# `image` must have a .numpy_image attribute (or use WorkflowImageData)
result = team_block.run(image=image, detections=detections, n_teams=2)
detections = result["detections"]  # detections.data["team_id"] is now set

# --- Track possession ---
possession_block = PossessionTrackerBlockV1()
result = possession_block.run(
    detections=detections,
    ball_class_id=1,
    ball_proximity_threshold=100.0,
)
print(result["possession_stats"])   # {0: 0.6, 1: 0.4}
print(result["possessing_team"])    # 0

# --- Compute distances ---
distance_block = DistanceCalculatorBlockV1()
result = distance_block.run(detections=detections)
print(result["detections"].data["distance"])  # cumulative px per tracker
```

### Example: Workflow JSON definition

```json
{
  "steps": [
    {
      "type": "sportvision/sports_detection_filter@v1",
      "name": "filter",
      "detections": "$steps.model.predictions"
    },
    {
      "type": "sportvision/team_classifier@v1",
      "name": "teams",
      "image": "$inputs.image",
      "detections": "$steps.filter.detections",
      "n_teams": 2
    },
    {
      "type": "sportvision/possession_tracker@v1",
      "name": "possession",
      "detections": "$steps.teams.detections",
      "ball_proximity_threshold": 100.0
    },
    {
      "type": "sportvision/distance_calculator@v1",
      "name": "distance",
      "detections": "$steps.teams.detections"
    }
  ]
}
```

## Try it on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohibShaikh/sportvision/blob/master/sportvision_colab.ipynb)

## Architecture

```
src/sportvision/
├── detection.py      # SportsDetector — wraps RF-DETR, maps COCO→sports classes
├── tracking.py       # SportsTracker — ByteTrack via supervision
├── teams.py          # TeamClassifier — KMeans on HSV jersey histograms
├── homography.py     # FieldHomography — pixel→field coords
├── analytics/
│   ├── possession.py # PossessionTracker — nearest-player-to-ball per frame
│   ├── speed.py      # SpeedEstimator — displacement/time → km/h
│   ├── distance.py   # DistanceCalculator — cumulative path length
│   └── heatmap.py    # HeatmapGenerator — 2D histogram + gaussian blur
├── annotators.py     # TeamColorAnnotator, StatsOverlayAnnotator, TrailAnnotator
├── pipeline.py       # SportVisionPipeline — orchestrates all modules
└── workflows/            # Roboflow Workflows plugin
    ├── _compat.py        # Inference compatibility shim
    ├── kinds.py          # Custom kind definitions
    ├── team_classifier/  # Team classification block
    ├── possession_tracker/   # Possession tracking block
    ├── distance_calculator/  # Distance calculation block
    └── sports_detection_filter/  # COCO→sports filter block
```

## Sports Class IDs

| ID | Class |
|----|-------|
| 0 | Player |
| 1 | Ball |
| 2 | Referee |
| 3 | Goalkeeper |

## Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| numpy | Arrays | Yes |
| opencv-python | Image processing, annotation | Yes |
| supervision | Detection/tracking data structures | Yes |
| scikit-learn | KMeans for team classification | Yes |
| pydantic | Workflow block manifests | Yes |
| inference | Roboflow Workflows engine | Optional (`[workflows]`) |
| rfdetr | Detection model | Optional (`[inference]`) |

## Development

```bash
git clone https://github.com/MohibShaikh/sportvision.git
cd sportvision
pip install -e ".[all]"

# Tests
pytest tests/ -v

# Lint
ruff check src/ tests/ && ruff format --check src/ tests/
```

## License

Apache-2.0
