# Evaluate before claiming accuracy

`basketball-provisional.json` contains three **provisional assistant annotations**, drawn from the raw cached basketball sample. They need independent human review. They are useful for exercising the evaluator, not establishing product accuracy. No video is redistributed here.

Generate a report with the same source and with white assigned to Team 1 (ID 0), black to Team 2 (ID 1):

```python
from sportvision.video import analyze_video
analyze_video(
    "test_data/basketball-1.mp4", "review.mp4", max_frames=150,
    team_colors=[(255, 255, 255), (21, 21, 21)],
)
```

```bash
python -m sportvision.evaluation review.json docs/evaluation/basketball-provisional.json
```

The source SHA-256 must match. Frame indices are zero-based; boxes are `[left, top, right, bottom]` in original image pixels. Human player IDs identify the same person across frames and are independent of model track IDs. Teams are 0 and 1.

- `ball_visible`: true/false scores whether at least one ball was detected on that frame. Null or omitted means not labeled. This is frame visibility, not box-level ball recall.
- `possession_team`: 0/1 is an assigned team; null explicitly labels no possession. Omit the field when uncertain.
- `players`: independently drawn boxes with `player_id` and `team_id`. The evaluator greedily matches predictions one-to-one at IoU >= 0.5, reports matched-box coverage and team accuracy, and counts track-ID changes between labeled matches. It is not an IDF1/HOTA benchmark.
- Empty denominators produce null, not a perfect or zero score.

Expand to diverse clips from the chosen fixed-camera basketball setup. Have a human review every label, reserve held-out clips, and report sample sizes and failures. Compare baseline and corrected results separately: manual correction is human-assisted performance, not automatic model accuracy. The downloadable report keeps measured team assignments alongside corrections.

Manual ball positions and confirmed possession remain separate from measured detections in the report. Evaluation output flags corrected reports; ball visibility metrics still describe the original detector. Possession scores on corrected reports measure assisted review, not automatic inference.


## Compare analysis modes

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/compare_analysis_modes.py \
  --source test_data/basketball-1.mp4 \
  --labels docs/evaluation/basketball-provisional.json \
  --output /tmp/analysis-modes.json
```

This processes every frame through the last label, preserving tracker history.
Both modes include the tracking repairs. The output records configuration,
uncorrected labeled-frame evidence, timing and metrics. Timings include startup
and sequential runs; use repeated warmed runs for performance comparisons.
Neither of the two ball-labeled frames is a negative example, so this diagnostic
cannot establish the false-positive rate. The 15%-of-height possession rule is
an uncalibrated heuristic, not a validated sports-specific model.

### Recorded diagnostic

`analysis-modes-provisional.json` records the local comparison. Detailed recovered
one of the two provisionally labeled visible-ball frames (frame 100), versus zero
in Quick. Replaying its uncorrected detections with body proximity assigned Team 2
on that frame. Frame 145 remained missed. Both modes matched all six labeled player
boxes, assigned five correctly, and had zero ID switches across four comparisons.
These are counts from a tiny development sample, not evidence of general accuracy.
