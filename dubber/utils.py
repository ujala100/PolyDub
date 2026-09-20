"""Small shared helpers used across every pipeline stage."""

from __future__ import annotations

import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def log(stage: str, message: str) -> None:
    """Print a single, consistently formatted progress line to the terminal.

    Every pipeline stage should report through this function so the user
    sees a clean, uniform stream of progress while the script runs, e.g.:

        [1/4 Download]     Fetching video metadata...
        [2/4 Transcribe]   Detected language: German
    """
    print(f"[{stage}] {message}", flush=True)


@contextmanager
def timed_stage(stage: str, message: str):
    """Context manager that times a stage and prints a start/finish line.

    Usage:
        with timed_stage("1/4 Download", "Downloading video"):
            do_the_download()
    """
    log(stage, f"{message}...")
    start = time.time()
    yield
    elapsed = time.time() - start
    log(stage, f"Done in {format_duration(elapsed)}.")


def format_duration(seconds: float) -> str:
    """Render a duration in seconds as e.g. '1h 04m 12s' or '38s'."""
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_timestamp(seconds: float) -> str:
    """Render seconds as HH:MM:SS.mmm, used for logging segment timing."""
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def ensure_empty_dir(path: Path) -> Path:
    """Create `path` fresh (removing it first if it already exists)."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_ffmpeg_available() -> None:
    """Fail fast with a clear message if ffmpeg/ffprobe aren't on PATH."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        joined = " and ".join(missing)
        sys.exit(
            f"Error: {joined} not found on PATH. Install ffmpeg "
            f"(https://ffmpeg.org/download.html) and make sure it's on "
            f"your PATH, then re-run this script."
        )
