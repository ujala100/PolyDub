"""Stage 4: Synthesize.

Converts each segment's translated English text into natural-sounding
speech using edge-tts (free, Microsoft neural voices). Each clip is then
time-stretched (sped up or slowed down) with ffmpeg's `atempo` filter so
its duration matches the original segment as closely as possible -- this
is what keeps the dub roughly in sync with the video instead of drifting
further out of sync as the video plays.

Both the TTS calls (network-bound) and the atempo stretching (CPU-bound,
one ffmpeg subprocess per segment) run concurrently rather than one at a
time -- for a long video with hundreds of segments, doing these
sequentially means most of the wall-clock time is just waiting on
network round-trips or process-launch overhead one at a time instead of
in parallel.
"""

from __future__ import annotations

import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import edge_tts

from .transcriber import Segment
from .utils import log

STAGE = "4/5 Synthesize"

# A natural-sounding neutral English voice. Swappable per run via CLI.
DEFAULT_VOICE = "en-US-AndrewNeural"

# atempo only guarantees good quality within roughly this range; beyond it
# speech becomes distorted, so we clamp and accept imperfect sync rather
# than mangle the audio.
_MIN_TEMPO = 0.6
_MAX_TEMPO = 1.7

# Standardized speech-quality output format for every fitted clip, so
# remixer.py can mix them with simple numpy slicing without huge RAM use.
_OUTPUT_SAMPLE_RATE = 24000

# How many TTS requests / ffmpeg processes to run at once. edge-tts is a
# free public endpoint -- too much concurrency risks throttling, so this
# stays modest rather than maxing out.
_MAX_CONCURRENT_TTS = 6
_MAX_CONCURRENT_FFMPEG = 4


async def _synthesize_one(text: str, voice: str, out_path: Path, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(out_path))


async def _synthesize_all(
    texts: list[str], voice: str, raw_paths: list[Path]
) -> None:
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_TTS)
    tasks = [
        _synthesize_one(text or ".", voice, path, semaphore)
        for text, path in zip(texts, raw_paths)
    ]
    await asyncio.gather(*tasks)


def _get_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _atempo_chain(factor: float) -> str:
    """Build an ffmpeg filter string for `factor`, chaining atempo filters
    if the ratio falls outside the single-filter's valid 0.5-2.0 range."""
    filters = []
    remaining = factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def _time_stretch(in_path: Path, out_path: Path, target_duration: float) -> float:
    """Stretch/compress `in_path` to approximately `target_duration` seconds,
    resampled to a standard mono 44.1kHz WAV. Returns the tempo factor
    actually applied (post-clamping)."""
    current_duration = _get_duration(in_path)

    base_cmd = ["ffmpeg", "-y", "-i", str(in_path), "-ac", "1", "-ar", str(_OUTPUT_SAMPLE_RATE)]

    if current_duration <= 0 or target_duration <= 0:
        subprocess.run(base_cmd + ["-loglevel", "error", str(out_path)], check=True)
        return 1.0

    factor = current_duration / target_duration
    clamped = max(_MIN_TEMPO, min(_MAX_TEMPO, factor))
    filter_str = _atempo_chain(clamped)

    subprocess.run(
        base_cmd + ["-filter:a", filter_str, "-loglevel", "error", str(out_path)],
        check=True,
    )
    return clamped


def _make_speech_chunks(
    segments: list[Segment], texts: list[str], max_chunk_seconds: float
) -> tuple[list[Segment], list[str]]:
    """Join nearby segments before TTS.

    Online TTS has meaningful per-request overhead.  A two-hour talk can
    otherwise create thousands of requests.  We retain the first/last
    timestamps, but make natural phrase-sized requests for contiguous speech.
    """
    chunks: list[Segment] = []
    chunk_texts: list[str] = []
    for segment, text in zip(segments, texts):
        clean = text.strip()
        if not clean:
            continue
        if chunks:
            previous = chunks[-1]
            gap = segment.start - previous.end
            proposed_duration = segment.end - previous.start
            proposed_length = len(chunk_texts[-1]) + len(clean) + 1
            if gap <= 0.45 and proposed_duration <= max_chunk_seconds and proposed_length <= 900:
                chunks[-1] = Segment(previous.start, segment.end, "")
                chunk_texts[-1] += " " + clean
                continue
        chunks.append(Segment(segment.start, segment.end, ""))
        chunk_texts.append(clean)
    return chunks, chunk_texts


def synthesize_segments(
    segments: list[Segment],
    english_texts: list[str],
    work_dir: Path,
    voice: str = DEFAULT_VOICE,
    max_chunk_seconds: float = 18.0,
) -> tuple[list[Segment], list[Path]]:
    """Generate one time-stretched English audio clip per segment.

    Returns a list of file paths, same length/order as `segments`, each
    clip's duration adjusted to fit its segment's original time slot.
    """
    chunks, chunk_texts = _make_speech_chunks(segments, english_texts, max_chunk_seconds)
    if not chunks:
        raise RuntimeError("No translated speech was available for synthesis.")
    raw_dir = work_dir / "tts_raw"
    fit_dir = work_dir / "tts_fit"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fit_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = [raw_dir / f"chunk_{i:04d}.mp3" for i in range(1, len(chunks) + 1)]
    fit_paths = [fit_dir / f"chunk_{i:04d}.wav" for i in range(1, len(chunks) + 1)]

    log(STAGE, f"Synthesizing {len(chunks)} speech chunks from {len(segments)} segments with voice '{voice}' "
               f"(up to {_MAX_CONCURRENT_TTS} concurrent requests)...")
    asyncio.run(_synthesize_all(chunk_texts, voice, raw_paths))
    log(STAGE, "All raw TTS clips generated; time-fitting each to its segment...")

    def _fit_one(args) -> float:
        i, seg, raw_path, fit_path = args
        return _time_stretch(raw_path, fit_path, seg.duration)

    jobs = list(zip(range(1, len(chunks) + 1), chunks, raw_paths, fit_paths))
    completed = 0
    with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_FFMPEG) as pool:
        for _ in pool.map(_fit_one, jobs):
            completed += 1
            if completed % 25 == 0 or completed == len(chunks):
                log(STAGE, f"  Time-fitted {completed}/{len(chunks)} chunks")

    return chunks, fit_paths
