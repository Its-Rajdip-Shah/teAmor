from pathlib import Path
from typing import Optional
import os
import subprocess
import sys

VIDEO_PATH = Path(__file__).with_name("Sad_Character_Animation_Request.mp4")
OUTPUT_DIR = Path(__file__).with_name("frames")


def extract_with_cv2(video_path: Path, out_dir: Path) -> int:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = out_dir / f"frame_{count:04d}.png"
        cv2.imwrite(str(out_path), frame)
        count += 1
    cap.release()
    return count


def extract_with_ffmpeg(video_path: Path, out_dir: Path) -> int:
    if not shutil_which("ffmpeg"):
        raise RuntimeError("ffmpeg not found and OpenCV is unavailable.")
    out_pattern = str(out_dir / "frame_%04d.png")
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vsync", "0", out_pattern]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("frame_*.png")))


def shutil_which(name: str) -> Optional[str]:
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def main() -> None:
    if not VIDEO_PATH.exists():
        print(f"❌ Video not found: {VIDEO_PATH}")
        sys.exit(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        count = extract_with_cv2(VIDEO_PATH, OUTPUT_DIR)
    except Exception:
        count = extract_with_ffmpeg(VIDEO_PATH, OUTPUT_DIR)
    print(f"✅ Done! Extracted {count} frames to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
