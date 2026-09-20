# Automated Video Dubbing System

Takes a YouTube URL in any spoken language and produces a new video dubbed
into natural-sounding English — same video, same rough timing/energy,
new audio.

## Pipeline

```
YouTube URL
    │  (1/5 Fetch)        yt-dlp
    ▼
source video (.mp4)
    │  (2/5 Transcribe)   ffmpeg (extract audio) + faster-whisper
    ▼
timestamped segments, original language
    │  (3/5 Translate)    deep-translator (Google Translate, free)
    ▼
timestamped segments, English text
    │  (4/5 Synthesize)   edge-tts + ffmpeg atempo (time-fit to segment)
    ▼
per-segment English audio clips, each ≈ matching its segment's duration
    │  (5/5 Remix)        pydub (timeline overlay) + ffmpeg (mux, -c:v copy)
    ▼
final dubbed video (.mp4)
```

Each stage lives in its own module under `dubber/` and only depends on
plain Python data (a list of `Segment(start, end, text)` objects, or a
list of English strings) going in and out — so any stage can be swapped
for a different implementation without touching the others. For example,
swapping the translator for IndicTrans2 on Hindi input, or the TTS engine
for a voice-cloning model, only requires changing that one file.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You also need **ffmpeg** installed and on your PATH:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: download from ffmpeg.org and add the `bin` folder to PATH

No API keys are required. Whisper runs locally (weights download once from
Hugging Face on first run), edge-tts and deep-translator use free public
endpoints.

## Usage

```bash
python main.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

Or run it with no argument and it will prompt you for a URL:

```bash
python main.py
```

### Useful options

```bash
python main.py "<url>" --model medium              # more accurate transcription (slower)
python main.py "<url>" --voice en-US-JennyNeural    # different English voice
python main.py "<url>" --out my_dub.mp4             # custom output path
python main.py "<url>" --device cuda                # use a GPU for Whisper, if you have one
python main.py "<url>" --keep-work-dir              # keep intermediate files for inspection
```

Run `python main.py --help` for the full list. Run `edge-tts --list-voices`
to see all available English voices.

The finished video is saved under `output/` by default, and progress for
every stage is printed to the terminal as it runs.

## Design notes / trade-offs

- **Timing alignment**: Whisper gives per-segment start/end timestamps.
  Each segment is translated and synthesized independently, then the
  resulting speech clip is time-stretched (`ffmpeg atempo`) to fit the
  original segment's duration as closely as possible, and placed on the
  output timeline at that segment's original start time. This is what
  keeps the dub in sync with the video rather than drifting as it
  progresses. Extreme stretch ratios (very fast talkers translated into
  much longer English text, or vice versa) are clamped to a 0.6x–1.7x
  range to avoid audibly distorted speech — beyond that range perfect
  sync is traded for natural-sounding audio.
- **Translation as its own stage**: rather than using Whisper's built-in
  translate mode, transcription (original language) and translation are
  kept as separate stages. This makes intermediate output inspectable
  (you can see exactly what was transcribed vs. what was translated) and
  makes it trivial to swap in a specialized translator (e.g. IndicTrans2
  for Hindi) without touching transcription code.
- **Video is never re-encoded**: the final mux uses `-c:v copy`, so visual
  quality is identical to the source; only the audio stream is replaced.
- **Multi-speaker dubbing / voice cloning** (diarization + per-speaker
  voices) was intentionally left out of the core pipeline — it's the
  assignment's stated stretch goal. The architecture is set up to support
  it later: `transcriber.py` would additionally tag each `Segment` with a
  speaker ID (via `pyannote.audio`), and `synthesizer.py` would pick a
  distinct voice (or a cloned voice via Coqui XTTS) per speaker ID instead
  of the single default voice.

## Known limitations

- Free translation and TTS endpoints (Google Translate, edge-tts) are
  rate-limited and can occasionally throttle on very long videos; the
  translator retries failed batches automatically.
- Whisper model download happens once on first run and requires internet
  access to Hugging Face.
