"""Manual smoke run on the locally cached basketball clip; no fabricated teams."""

from examples.demo import analyze_video

if __name__ == "__main__":
    print(
        analyze_video(
            "test_data/basketball-1.mp4", "test_data/output.mp4", max_frames=150
        )
    )
