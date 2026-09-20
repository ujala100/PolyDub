"""Stage 3: Translate.

Translates each transcribed segment's text into natural, meaning-preserving
English (not a literal word-for-word conversion). Uses deep-translator's
free Google Translate backend, which handles arbitrary source languages
without needing an API key.

For speed on long videos with many segments, several segments' text is
joined into a single request (newline-delimited) rather than sending one
request per segment -- for a video with hundreds of segments, this cuts
the number of network round-trips by roughly the batch size. If a
batch's translated result doesn't split back into the same number of
lines (the service occasionally merges/reflows short lines), that batch
is retried segment-by-segment instead, so correctness never depends on
the batching working perfectly.

If the service is still unavailable after retrying, this raises
TranslationServiceUnavailable so the caller can fall back to a fully
local alternative (see transcriber.transcribe_translate_fallback).
"""

from __future__ import annotations

import random
import time

from deep_translator import GoogleTranslator

from .transcriber import Segment
from .utils import log

STAGE = "3/5 Translate"

_MAX_RETRIES = 4
_BASE_BACKOFF = 2.0            # seconds; doubles each retry, plus jitter
_PACE_BETWEEN_REQUESTS = 0.25  # seconds between successive requests

# Segments per batched request. The free endpoint has a per-request size
# cap, so batches are also bounded by character count.
_BATCH_SIZE = 15
_BATCH_CHAR_LIMIT = 3500
_LINE_SEP = "\n"


class TranslationServiceUnavailable(RuntimeError):
    """Raised when the external translation service can't be reached
    after retrying. Callers may fall back to a local alternative."""


def _chunk_segments(segments: list[Segment]) -> list[list[Segment]]:
    batches: list[list[Segment]] = []
    current: list[Segment] = []
    current_len = 0
    for seg in segments:
        seg_len = len(seg.text) + 1
        if current and (len(current) >= _BATCH_SIZE or current_len + seg_len > _BATCH_CHAR_LIMIT):
            batches.append(current)
            current, current_len = [], 0
        current.append(seg)
        current_len += seg_len
    if current:
        batches.append(current)
    return batches


def _translate_with_retry(translator: GoogleTranslator, text: str, label: str) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return translator.translate(text)
        except Exception as exc:  # rate limits / network hiccups
            last_exc = exc
            if attempt == _MAX_RETRIES:
                break
            backoff = _BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 1)
            log(STAGE, f"  {label} failed ({exc}); retrying in {backoff:.1f}s ({attempt}/{_MAX_RETRIES})...")
            time.sleep(backoff)
    raise TranslationServiceUnavailable(
        f"Translation service unavailable after {_MAX_RETRIES} attempts on {label}: {last_exc}"
    )


def translate_segments(
    segments: list[Segment], source_language: str, target_language: str = "en"
) -> list[str]:
    """Return a list of English strings, one per input segment, same order.

    Raises TranslationServiceUnavailable if the service keeps failing.
    """
    if source_language in ("en", "english"):
        log(STAGE, "Source is already English; skipping translation.")
        return [seg.text for seg in segments]

    translator = GoogleTranslator(source="auto", target=target_language)
    batches = _chunk_segments(segments)
    log(STAGE, f"Translating {len(segments)} segments in {len(batches)} batch(es)...")

    translated: list[str] = []
    for b_idx, batch in enumerate(batches, start=1):
        joined = _LINE_SEP.join(seg.text for seg in batch)
        result_text = _translate_with_retry(translator, joined, f"batch {b_idx}/{len(batches)}")
        lines = result_text.split(_LINE_SEP)

        if len(lines) == len(batch):
            translated.extend(lines)
        else:
            # Service reflowed the lines (merged/split them); fall back to
            # translating this batch's segments one at a time so output
            # stays correctly aligned with the input segments.
            log(
                STAGE,
                f"  Batch {b_idx} came back as {len(lines)} lines for "
                f"{len(batch)} segments; retranslating individually...",
            )
            for seg in batch:
                translated.append(_translate_with_retry(translator, seg.text, "segment (batch fallback)"))
                time.sleep(_PACE_BETWEEN_REQUESTS)

        if b_idx <= 2 or b_idx == len(batches):
            preview_orig = batch[0].text[:50]
            preview_en = translated[len(translated) - len(batch)][:50]
            log(STAGE, f"  [batch {b_idx}/{len(batches)}] '{preview_orig}' -> '{preview_en}'")

        if b_idx < len(batches):
            time.sleep(_PACE_BETWEEN_REQUESTS)

    return translated
