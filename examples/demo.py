"""Command-line example; the packaged runner also powers the browser app."""

import argparse
import json

from sportvision.video import analyze_video


def main():
    parser = argparse.ArgumentParser(description="Annotate a sports clip")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--analysis-mode", choices=["quick", "detailed"], default="quick"
    )
    args = parser.parse_args()
    print(json.dumps(analyze_video(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
