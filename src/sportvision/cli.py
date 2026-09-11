"""Installed entry point for the local browser app."""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="SportVision, review a sports clip")
    parser.add_argument("command", choices=["app"], nargs="?", default="app")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if importlib.util.find_spec("streamlit") is None:
        parser.exit(1, 'Install the app first: pip install "sportvision[app]"\n')
    # Imported here because theme imports streamlit, which may be absent above.
    from sportvision.theme import PALETTE

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
                # Base widget chrome has to match theme.py, otherwise Streamlit
                # paints its own defaults underneath the stylesheet.
                "--theme.base=dark",
                f"--theme.primaryColor={PALETTE['accent']}",
                f"--theme.backgroundColor={PALETTE['bg']}",
                f"--theme.secondaryBackgroundColor={PALETTE['bg_2']}",
                f"--theme.textColor={PALETTE['fg']}",
            ]
        )
    )
