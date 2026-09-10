"""Installed entry point for the local browser app."""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="SportVision — review a sports clip")
    parser.add_argument("command", choices=["app"], nargs="?", default="app")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if importlib.util.find_spec("streamlit") is None:
        parser.exit(1, 'Install the app first: pip install "sportvision[app]"\n')
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(Path(__file__).with_name("app.py")),
                "--server.address=127.0.0.1",
                "--server.headless=true",
                f"--server.port={args.port}",
                "--browser.gatherUsageStats=false",
            ]
        )
    )
