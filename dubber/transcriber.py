"""Stage 2: Transcribe.

Extracts the audio track from the source video and runs it through
faster-whisper (a fast, CTranslate2-based re-implementation of OpenAI's
Whisper) to get a list of timestamped segments in the original language.

We deliberately transcribe in the *original* language here rather than
using Whisper's built-in "translate" task, and do translation as its own
explicit stage (see translator.py). Keeping the two separate makes each
stage independently inspectable/debuggable and lets us swap in a
higher-quality translator (e.g. IndicTrans2 for Indian languages) later
without touching the transcription code.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from .utils import format_timestamp, log

STAGE = "2/5 Transcribe"

# faster-whisper's feature extractor builds a spectrogram for the complete
# input file before decoding it.  Keeping each input at ten minutes prevents
# multi-hour videos from needing several gigabytes of temporary RAM.
_AUDIO_CHUNK_SECONDS = 60


@dataclass
class Segment:
    """One chunk of speech: a time range plus its text."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    """Extract a 16kHz mono WAV track from `video_path` using ffmpeg.

    Whisper models expect 16kHz mono audio; doing this conversion once
    up front (rather than letting a library guess) keeps behavior
    predictable and fast.
    """
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-loglevel", "error",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)
    return audio_path


def split_audio(audio_path: Path, chunks_dir: Path) -> list[Path]:
    """Split a WAV into independently transcribable, memory-safe chunks."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pattern = chunks_dir / "part_%05d.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path), "-map", "0:a:0",
            "-f", "segment", "-segment_time", str(_AUDIO_CHUNK_SECONDS),
            "-reset_timestamps", "1", "-c:a", "pcm_s16le", "-loglevel", "error",
            str(pattern),
        ],
        check=True,
    )
    parts = sorted(chunks_dir.glob("part_*.wav"))
    if not parts:
        raise RuntimeError("ffmpeg did not produce any audio chunks for transcription.")
    return parts


# Segments with a no_speech_prob above this are almost always hallucinated
# text over music/silence/noise rather than real speech, and get dropped.
_NO_SPEECH_PROB_THRESHOLD = 0.6


def _load_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        except Exception:
            device = "cpu"
    if device == "cuda" and compute_type == "int8":
        compute_type = "float16"
    log(STAGE, f"Loading Whisper model '{model_size}' ({device}/{compute_type})...")
    # On CPU, faster-whisper defaults to a conservative thread count unless
    # told otherwise; explicitly using all available cores can meaningfully
    # cut transcription time on multi-core machines.
    cpu_threads = os.cpu_count() or 4
    kwargs = {"cpu_threads": cpu_threads} if device == "cpu" else {}
    return WhisperModel(model_size, device=device, compute_type=compute_type, **kwargs)


def transcribe(
    video_path: Path,
    work_dir: Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
) -> tuple[list[Segment], str]:
    """Return (segments, detected_language_name) for `video_path`.

    model_size: one of tiny/base/small/medium/large-v3. Bigger = more
    accurate but slower; "small" is a fast default, but "medium" or
    "large-v3" is recommended for graded/final output -- smaller models
    are more prone to hallucinating text over music or unclear audio.
    beam_size: Whisper's search width. Lower (e.g. 1) is meaningfully
    faster on CPU with a small accuracy cost; default 5 favors accuracy.
    """
    audio_path = work_dir / "audio.wav"
    log(STAGE, "Extracting audio track from video...")
    extract_audio(video_path, audio_path)

    model = _load_model(model_size, device, compute_type)

    segments: list[Segment] = []
    dropped = 0
    chunks = split_audio(audio_path, work_dir / "audio_chunks")
    log(STAGE, f"Transcribing {len(chunks)} audio chunk(s) of up to {_AUDIO_CHUNK_SECONDS} seconds each...")
    language_name = "unknown"
    processed = 0
    for chunk_index, chunk_path in enumerate(chunks):
        # The segment muxer resets each chunk to zero, so restore its global
        # position.  This makes later synthesis/remixing line up as before.
        offset = chunk_index * _AUDIO_CHUNK_SECONDS
        raw_segments, info = model.transcribe(
            str(chunk_path), task="transcribe", vad_filter=True,
            word_timestamps=False, beam_size=beam_size,
        )
        if chunk_index == 0 and info:
            language_name = info.language
        for seg in raw_segments:
            processed += 1
            text = seg.text.strip()
            if not text:
                continue
            no_speech_prob = getattr(seg, "no_speech_prob", 0.0)
            if no_speech_prob > _NO_SPEECH_PROB_THRESHOLD:
                dropped += 1
                continue
            segments.append(Segment(start=seg.start + offset, end=seg.end + offset, text=text))
            if processed % 25 == 0:
                log(STAGE, f"  Processed {processed} speech segments (currently {format_timestamp(seg.end + offset)})")
        log(STAGE, f"  Finished audio chunk {chunk_index + 1}/{len(chunks)}")

    log(
        STAGE,
        f"Detected language: {language_name} ({len(segments)} segments kept"
        + (f", {dropped} dropped as likely hallucinated" if dropped else "") + ")",
    )
    if model_size in ("tiny", "base", "small"):
        log(
            STAGE,
            f"Note: using the '{model_size}' model. For your graded submission "
            f"videos, consider --model medium (or large-v3) for higher transcription "
            f"accuracy, especially on non-English audio.",
        )

    if not segments:
        raise RuntimeError(
            "No speech detected in the video's audio track. "
            "Check that the source video actually contains spoken audio."
        )

    return segments, language_name


def transcribe_translate_fallback(
    work_dir: Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
) -> list[Segment]:
    """Fallback path: get English segments directly from Whisper's own
    built-in translate task, entirely locally (no external translation
    service). Used when the external translator (translator.py) is
    unavailable, e.g. rate-limited or offline.

    Reuses the audio already extracted by `transcribe()` in the same
    work_dir, so this is a cheap second pass, not a full re-run.
    """
    audio_path = work_dir / "audio.wav"
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Expected extracted audio at {audio_path}; run transcribe() first."
        )

    model = _load_model(model_size, device, compute_type)
    log(STAGE, "Running Whisper's local translate mode as a fallback (no external service needed)...")
    segments: list[Segment] = []
    chunks = split_audio(audio_path, work_dir / "audio_chunks_translate")
    for chunk_index, chunk_path in enumerate(chunks):
        raw_segments, _info = model.transcribe(
            str(chunk_path), task="translate", vad_filter=True,
            word_timestamps=False, beam_size=beam_size,
        )
        offset = chunk_index * _AUDIO_CHUNK_SECONDS
        for seg in raw_segments:
            text = seg.text.strip()
            if not text or getattr(seg, "no_speech_prob", 0.0) > _NO_SPEECH_PROB_THRESHOLD:
                continue
            segments.append(Segment(start=seg.start + offset, end=seg.end + offset, text=text))
        log(STAGE, f"  Finished translated audio chunk {chunk_index + 1}/{len(chunks)}")

    if not segments:
        raise RuntimeError("Whisper translate fallback produced no usable segments either.")

    return segments
