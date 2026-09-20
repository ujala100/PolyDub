"""Stage 1: Fetch.

Downloads a video from a YouTube URL using yt-dlp. yt-dlp is used instead
of pytube because it's actively maintained and far more resilient to
YouTube's frequent player/API changes.
"""

from __future__ import annotations

from pathlib import Path

import yt_dlp

from .utils import log

STAGE = "1/5 Fetch"


def download_video(url: str, out_dir: Path) -> Path:
    """Download `url` into `out_dir` and return the path to the video file.

    We ask yt-dlp for a single progressive-friendly stream: best video +
    best audio, merged into one .mp4 by ffmpeg (which yt-dlp shells out to
    automatically). Downloading video+audio together (rather than just the
    audio) is required here because we need the original video stream
    untouched for the final remix step.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "source.%(ext)s")

    last_percent = {"value": -10}  # throttle progress spam to every ~10%

    def _progress_hook(d: dict) -> None:
        if d["status"] == "downloading":
            percent_str = d.get("_percent_str", "").strip().replace("%", "")
            try:
                percent = float(percent_str)
            except ValueError:
                return
            if percent - last_percent["value"] >= 10:
                last_percent["value"] = percent
                log(STAGE, f"Downloading... {percent:.0f}%")
        elif d["status"] == "finished":
            log(STAGE, "Download finished, merging streams if needed...")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_progress_hook],
        # Be persistent about transient network drops (connection resets,
        # timeouts) rather than giving up after yt-dlp's low defaults --
        # these are common on flaky Wi-Fi/VPNs and usually resolve on retry.
        "retries": 15,
        "fragment_retries": 15,
        "socket_timeout": 30,
    }

    log(STAGE, f"Fetching video info for {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "video")
        duration = info.get("duration")

    # yt-dlp creates files such as ``source.f616.mp4.part-Frag785.part``
    # while downloading DASH streams.  They match source.* too, but are not
    # valid videos and must never be passed to ffmpeg.  Prefer the known
    # merged output, then fall back to another finalized media file.
    preferred = out_dir / "source.mp4"
    media_extensions = {".mp4", ".mkv", ".webm", ".m4v"}
    finalized = [
        path for path in out_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in media_extensions
        and ".part" not in path.name.lower()
        and ".ytdl" not in path.name.lower()
    ]
    if preferred.exists():
        video_path = preferred
    elif finalized:
        video_path = max(finalized, key=lambda path: path.stat().st_size)
    else:
        fragments = list(out_dir.glob("*.part"))
        detail = "only incomplete download fragments remain" if fragments else "no output file was found"
        raise RuntimeError(f"yt-dlp reported success but {detail}.")

    duration_str = f"{duration}s" if duration else "unknown length"
    log(STAGE, f"Downloaded '{title}' ({duration_str}) -> {video_path.name}")
    return video_path
