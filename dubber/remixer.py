"""Stage 5: Remix & Output.

Places every synthesized (and time-fitted) segment clip at its original
timestamp on a single audio timeline the length of the whole video, then
swaps that track into the source video in place of the original audio.

Implementation note: this builds the timeline as a disk-backed numpy int16 array
and writes each clip directly into its slice, rather than looping
pydub's `overlay()`. pydub's AudioSegment is immutable, so calling
overlay() in a loop copies the *entire* buffer built so far on every
single call -- fine for a few segments, but for a long video with
hundreds of segments that turns into hundreds of full-buffer copies
(cost scales with segments x video length, not just video length). The
numpy version only touches the samples each clip actually occupies, so
cost scales with total speech duration, however many segments there are.

The video stream is copied (`-c:v copy`), never re-encoded, so visual
quality is identical to the source and the swap is fast.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

from .transcriber import Segment
from .utils import log

STAGE = "5/5 Remix"

SAMPLE_RATE = 24000  # must match synthesizer.py's fitted-clip output rate


def _get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _read_mono_int16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(
                f"{path} is {w.getframerate()}Hz/{w.getnchannels()}ch/"
                f"{w.getsampwidth()*8}bit; expected {SAMPLE_RATE}Hz mono 16-bit. "
                f"(synthesizer.py should already produce clips in this format)"
            )
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16)


def build_dubbed_track(
    segments: list[Segment],
    clip_paths: list[Path],
    video_path: Path,
    work_dir: Path,
) -> Path:
    """Overlay every dubbed clip onto a silent track at its original start
    time, and return the path to the resulting single audio file."""
    total_duration_s = _get_video_duration(video_path) + 1.0
    total_samples = int(total_duration_s * SAMPLE_RATE)
    log(STAGE, f"Building {total_duration_s:.1f}s dubbed audio timeline ({len(segments)} clips)...")

    # A 2-hour 44.1kHz timeline used to occupy >600 MB of RAM.  A disk-backed
    # 24kHz timeline is speech-quality appropriate and makes long videos safe
    # on ordinary machines without slowing down the video stream itself.
    raw_timeline = work_dir / "dubbed_audio.pcm"
    timeline = np.memmap(raw_timeline, dtype=np.int16, mode="w+", shape=(total_samples,))
    timeline[:] = 0

    for i, (seg, clip_path) in enumerate(zip(segments, clip_paths), start=1):
        clip = _read_mono_int16(clip_path)
        start_sample = int(seg.start * SAMPLE_RATE)
        end_sample = start_sample + len(clip)

        if end_sample > total_samples:
            # Clip runs past the end of the video (can happen if the last
            # segment's timing is right at the edge); trim rather than crash.
            clip = clip[: total_samples - start_sample]
            end_sample = total_samples

        # Segments shouldn't overlap in time, but add (with clipping to the
        # valid int16 range) rather than overwrite, in case of rounding.
        mixed = timeline[start_sample:end_sample].astype(np.int32) + clip.astype(np.int32)
        timeline[start_sample:end_sample] = np.clip(mixed, -32768, 32767).astype(np.int16)

        if i % 25 == 0 or i == len(segments):
            log(STAGE, f"  Placed {i}/{len(segments)} clips onto timeline")

    timeline.flush()
    del timeline
    track_path = work_dir / "dubbed_audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-i", str(raw_timeline), "-loglevel", "error", str(track_path),
    ], check=True)
    raw_timeline.unlink(missing_ok=True)

    return track_path


def mux(video_path: Path, dubbed_audio_path: Path, output_path: Path) -> Path:
    """Replace `video_path`'s audio with `dubbed_audio_path`, video untouched."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(STAGE, "Muxing dubbed audio into the original video (video re-encode skipped)...")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(dubbed_audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-loglevel", "error",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    log(STAGE, f"Final dubbed video saved to: {output_path}")
    return output_path
