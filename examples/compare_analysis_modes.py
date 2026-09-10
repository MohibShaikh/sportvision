"""Reproduce a partial-label comparison without manual corrections.

See docs/evaluation/README.md for the reproduction command.
"""

import argparse
import json
import tempfile
from pathlib import Path

from sportvision.evaluation import evaluate
from sportvision.video import analyze_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labels = json.loads(Path(args.labels).read_text())
    comparisons = {}
    with tempfile.TemporaryDirectory(prefix="sportvision-eval-") as tmp:
        for mode in ("quick", "detailed"):
            report = analyze_video(
                args.source,
                Path(tmp) / f"{mode}.mp4",
                max_frames=1 + max(row["frame_index"] for row in labels["frames"]),
                team_colors=[(255, 255, 255), (21, 21, 21)],
                analysis_mode=mode,
            )
            comparisons[mode] = {
                "settings": {
                    key: report[key]
                    for key in (
                        "model",
                        "infer_size",
                        "confidence_threshold",
                        "ball_confidence_threshold",
                        "tracking_minimum_iou",
                        "possession_method",
                        "frames_processed",
                        "processing_time_s",
                    )
                },
                "metrics": evaluate(report["review"], labels),
                "labeled_frame_evidence": [
                    row
                    for row in report["review"]["frames"]
                    if row["frame_index"]
                    in {label["frame_index"] for label in labels["frames"]}
                ],
            }
            print(f"{mode}: {json.dumps(comparisons[mode]['metrics'])}", flush=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "clip_sha256": labels["clip_sha256"],
                "annotation_status": labels.get("annotation_status"),
                "note": (
                    "Both modes include tracking fixes. Three provisional frames "
                    "cannot establish production accuracy or false-positive rates. "
                    "Timing includes model startup; modes run sequentially."
                ),
                "comparisons": comparisons,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
