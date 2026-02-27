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
pip install "sportvision[all] @ git+https://github.com/MohibShaikh/sportvision.git"
```

```python
from sportvision.pipeline import SportVisionPipeline

pipeline = SportVisionPipeline()
result = pipeline.process_frame(frame)
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
└── workflows/
    └── blocks.py     # Roboflow Workflow block stubs
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
