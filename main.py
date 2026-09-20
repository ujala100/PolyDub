#!/usr/bin/env python3
"""
Automated Video Dubbing System
===============================

Takes a YouTube URL (any spoken language) and produces a new video dubbed
into natural-sounding English, with the original visuals untouched.

Usage:
    python main.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"
    python main.py "https://youtu.be/XXXXXXXXXXX" --model small --voice en-US-JennyNeural
    python main.py                                   # prompts for a URL interactively

Run `python main.py --help` for all options.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dubber import downloader, remixer, synthesizer, transcriber, translator
from dubber.utils import check_ffmpeg_available, ensure_empty_dir, format_duration, log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dub a YouTube video into English, preserving video and timing.",
    )
    parser.add_argument(
        "url", nargs="?", default=None,
        help="YouTube video URL. If omitted, you'll be prompted for one.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Path for the final dubbed .mp4 (default: output/<video_id>_dubbed.mp4)",
    )
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size for transcription (default: small). "
             "Bigger = more accurate, slower. 'base' is the fast default; use "
             "small/medium for a final-quality render.",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Device for Whisper (default: auto, which uses CUDA when available).",
    )
    parser.add_argument(
        "--beam-size", type=int, default=1,
        help="Whisper search width (default: 5). Lower (e.g. 1) transcribes "
             "noticeably faster on CPU at a small accuracy cost.",
    )
    parser.add_argument(
        "--tts-chunk-seconds", type=float, default=18.0,
        help="Merge adjacent subtitle segments into speech chunks up to this duration "
             "before calling TTS (default: 18). Larger is much faster for long videos.",
    )
    parser.add_argument(
        "--voice", default=synthesizer.DEFAULT_VOICE,
        help=f"edge-tts voice name for the dubbed speech (default: {synthesizer.DEFAULT_VOICE}). "
             "Run `edge-tts --list-voices` to see all options.",
    )
    parser.add_argument(
        "--work-dir", default=None,
        help="Directory for intermediate files (default: a temp folder under .cache/).",
    )
    parser.add_argument(
        "--keep-work-dir", action="store_true",
        help="Don't delete intermediate files (audio, per-segment clips) after finishing.",
    )
    return parser.parse_args()


def get_url_from_args_or_prompt(args: argparse.Namespace) -> str:
    if args.url:
        return args.url.strip()
    url = input("Enter a YouTube URL to dub: ").strip()
    if not url:
        sys.exit("Error: no URL provided.")
    return url


def main() -> None:
    args = parse_args()
    if args.tts_chunk_seconds <= 0:
        sys.exit("Error: --tts-chunk-seconds must be greater than zero.")
    check_ffmpeg_available()

    url = get_url_from_args_or_prompt(args)

    project_root = Path(__file__).resolve().parent
    work_dir = Path(args.work_dir) if args.work_dir else project_root / ".cache" / f"job_{int(time.time())}"
    ensure_empty_dir(work_dir)

    overall_start = time.time()
    log("START", f"Dubbing job started for: {url}")
    log("START", f"Intermediate files: {work_dir}")

    try:
        # Stage 1: Fetch
        stage_start = time.time()
        video_path = downloader.download_video(url, work_dir / "source")
        log("1/5 Fetch", f"Completed in {format_duration(time.time() - stage_start)}")

        # Stage 2: Transcribe
        stage_start = time.time()
        segments, language = transcriber.transcribe(
            video_path, work_dir, model_size=args.model, device=args.device,
            beam_size=args.beam_size,
        )
        log("2/5 Transcribe", f"Completed in {format_duration(time.time() - stage_start)}")

        # Stage 3: Translate (falls back to Whisper's local translate mode
        # if the external translation service is rate-limited/unreachable)
        stage_start = time.time()
        try:
            english_texts = translator.translate_segments(segments, source_language=language)
        except translator.TranslationServiceUnavailable as exc:
            log(
                "3/5 Translate",
                f"External translation unavailable ({exc}). "
                f"Falling back to Whisper's local translate mode...",
            )
            segments = transcriber.transcribe_translate_fallback(
                work_dir, model_size=args.model, device=args.device, beam_size=args.beam_size,
            )
            english_texts = [seg.text for seg in segments]
        log("3/5 Translate", f"Completed in {format_duration(time.time() - stage_start)}")

        # Stage 4: Synthesize
        stage_start = time.time()
        speech_chunks, clip_paths = synthesizer.synthesize_segments(
            segments, english_texts, work_dir, voice=args.voice,
            max_chunk_seconds=args.tts_chunk_seconds,
        )
        log("4/5 Synthesize", f"Completed in {format_duration(time.time() - stage_start)}")

        # Stage 5: Remix & Output
        stage_start = time.time()
        dubbed_track = remixer.build_dubbed_track(speech_chunks, clip_paths, video_path, work_dir)

        if args.out:
            output_path = Path(args.out)
        else:
            output_path = project_root / "output" / f"{video_path.stem}_dubbed.mp4"
        remixer.mux(video_path, dubbed_track, output_path)
        log("5/5 Remix", f"Completed in {format_duration(time.time() - stage_start)}")

    except Exception as exc:
        log("ERROR", str(exc))
        raise
    finally:
        if not args.keep_work_dir and work_dir.exists():
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

    elapsed = time.time() - overall_start
    log("DONE", f"Total processing time: {format_duration(elapsed)}")
    log("DONE", f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
