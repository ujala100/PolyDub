##🎙️ Automated Video Dubbing System

> **Turn any spoken-language YouTube video into a naturally voiced English version — while preserving the original video, timing, and conversational flow.**

An end-to-end **multilingual AI video dubbing pipeline** that downloads a YouTube video, detects and transcribes speech in its original language, translates the transcript into natural English, synthesizes English speech, automatically fits the generated audio to the original timestamps, and produces a final dubbed video without re-encoding the original visuals.

The system is designed as a **modular, replaceable AI pipeline**: transcription, translation, synthesis, and audio remixing are independent stages that communicate through simple Python data structures.

---

## ✨ Why This Project?

Most simple dubbing pipelines treat translation and speech synthesis as one monolithic process.

This project takes a different approach:

**Video → Speech → Meaning → English Speech → Time-Aligned Audio → Final Video**

Each stage is independently replaceable.

That means the same architecture can evolve from:

* Whisper → IndicTrans2
* Google Translate → a domain-specific translation model
* Edge TTS → XTTS / voice cloning
* Single-speaker dubbing → speaker diarization
* Local execution → GPU/cloud inference
* Script → production-grade media processing service

The goal is not only to make a working dub, but to demonstrate how an **AI system can be decomposed into independently testable and replaceable components.**

---

# 🚀 Demo Pipeline

```text
                     AUTOMATED VIDEO DUBBING SYSTEM

 ┌─────────────────┐
 │  YouTube URL    │
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  1. FETCH       │
 │     yt-dlp      │
 └────────┬────────┘
          │
          ▼
      source.mp4
          │
          ▼
 ┌─────────────────┐
 │  2. TRANSCRIBE  │
 │  FFmpeg         │
 │  faster-whisper │
 └────────┬────────┘
          │
          ▼
 ┌────────────────────────────┐
 │ Timestamped Segments       │
 │                            │
 │ [00:00 - 00:04] नमस्ते...  │
 │ [00:04 - 00:08] आज हम...   │
 └────────────┬───────────────┘
              │
              ▼
 ┌─────────────────┐
 │  3. TRANSLATE   │
 │  deep-translator│
 │  Google Translate│
 └────────┬────────┘
          │
          ▼
 ┌────────────────────────────┐
 │ English Segments            │
 │                            │
 │ [00:00 - 00:04] Hello...    │
 │ [00:04 - 00:08] Today...    │
 └────────────┬───────────────┘
              │
              ▼
 ┌─────────────────┐
 │  4. SYNTHESIZE  │
 │     edge-tts    │
 │     FFmpeg      │
 └────────┬────────┘
          │
          ▼
 ┌────────────────────────────┐
 │ Time-fitted English Audio  │
 └────────────┬───────────────┘
              │
              ▼
 ┌─────────────────┐
 │  5. REMIX       │
 │  pydub + FFmpeg │
 │  -c:v copy      │
 └────────┬────────┘
          │
          ▼
 ┌──────────────────────────────┐
 │      🎬 FINAL DUBBED VIDEO   │
 │           .mp4               │
 └──────────────────────────────┘
```

---

# 🎯 Core Features

### 🌍 Multilingual Speech Recognition

Uses **faster-whisper** to transcribe speech while preserving segment-level timestamps.

```text
Input:
Video in Hindi / Marathi / Spanish / French / ...

             ↓

Whisper

             ↓

[
  Segment(start=0.0, end=4.2, text="..."),
  Segment(start=4.2, end=8.7, text="..."),
  ...
]
```

This timestamp information becomes the foundation for synchronization throughout the pipeline.

---

### 🧠 Meaning-Preserving Translation

Translation is deliberately separated from transcription.

Instead of asking Whisper to directly output English, the pipeline performs:

```text
Original Speech
      ↓
Original-language transcript
      ↓
Translation layer
      ↓
Natural English transcript
```

This makes the translation component independently replaceable.

For example:

```text
Current:
deep-translator / Google Translate

Possible upgrade:
IndicTrans2
NLLB
MarianMT
LLM-based translation
Domain-specific translation models
```

This is particularly useful for Indian-language → English translation workflows.

---

### 🗣️ Natural English Speech

English text is converted into speech using **Edge TTS**.

The synthesis layer is intentionally isolated so that alternative TTS systems can be introduced without changing the rest of the pipeline.

Potential future implementations include:

* Coqui XTTS
* voice cloning models
* local neural TTS
* speaker-specific synthesis
* custom voice profiles

---

### ⏱️ Automatic Timing Alignment

One of the central engineering challenges is that:

> **Translated English speech does not necessarily have the same duration as the original speech.**

For every segment:

```text
Original segment duration
          ↓
English TTS generation
          ↓
Generated duration
          ↓
Calculate required adjustment
          ↓
FFmpeg atempo
          ↓
Time-fitted audio
```

The generated audio is placed at the original segment's timestamp.

Extreme time-stretching is intentionally constrained to approximately:

```text
0.6x ───────────── 1.0x ───────────── 1.7x
```

This creates a deliberate trade-off:

> **Natural-sounding speech is prioritized over mathematically perfect synchronization when the translated sentence is substantially longer or shorter than the original.**

---

### 🎬 Video-Preserving Output

The original video stream is not re-encoded.

FFmpeg uses:

```bash
-c:v copy
```

Therefore:

* Original visual quality is preserved
* Video encoding time is reduced
* CPU/GPU resources are not wasted re-encoding frames
* Only the audio stream is replaced

---

# 🏗️ Architecture

The project follows a modular pipeline architecture.

```text
dubber/
│
├── downloader.py
│       │
│       └── YouTube → MP4
│
├── transcriber.py
│       │
│       └── MP4 → timestamped Segments
│
├── translator.py
│       │
│       └── Segments → English text
│
├── synthesizer.py
│       │
│       └── English text → audio clips
│
├── remixer.py
│       │
│       └── audio clips → timeline → final video
│
└── models.py
        │
        └── Shared data structures
```

### Pipeline Contract

The modules communicate through simple data structures rather than tightly coupled implementations.

Conceptually:

```python
Segment(
    start: float,
    end: float,
    text: str
)
```

Therefore:

```text
Downloader
     ↓
Transcriber
     ↓
Translator
     ↓
Synthesizer
     ↓
Remixer
```

can evolve independently.

---

# 🔌 Replaceable Components

The architecture intentionally avoids locking the entire application to a single AI model.

| Stage             | Current Implementation | Possible Upgrade            |
| ----------------- | ---------------------- | --------------------------- |
| Video Fetching    | yt-dlp                 | Custom media ingestion      |
| Audio Extraction  | FFmpeg                 | FFmpeg / PyAV               |
| Transcription     | faster-whisper         | Whisper / WhisperX          |
| Translation       | deep-translator        | IndicTrans2 / NLLB / LLM    |
| TTS               | Edge TTS               | XTTS / local neural TTS     |
| Time Fitting      | FFmpeg atempo          | Advanced duration control   |
| Audio Remix       | pydub                  | FFmpeg / PyAV               |
| Speaker Detection | —                      | pyannote.audio              |
| Voice Assignment  | Single voice           | Speaker-specific voices     |
| Voice Cloning     | —                      | XTTS / other cloning models |

This makes the project suitable as a foundation for a more advanced **AI localization system**.

---

# 📁 Project Structure

```text
automated-video-dubbing/
│
├── dubber/
│   ├── __init__.py
│   ├── downloader.py
│   ├── transcriber.py
│   ├── translator.py
│   ├── synthesizer.py
│   ├── remixer.py
│   └── models.py
│
├── output/
│   └── .gitkeep
│
├── work/
│   └── .gitkeep
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Technology Stack

### AI / ML

* **faster-whisper** — multilingual automatic speech recognition
* **Whisper** — speech recognition foundation
* **Google Translate endpoint via deep-translator** — translation
* **Edge TTS** — neural text-to-speech

### Media Processing

* **FFmpeg** — audio extraction, time stretching and final muxing
* **pydub** — timeline-based audio composition
* **yt-dlp** — reliable YouTube media downloading

### Engineering

* Python
* Modular pipeline architecture
* CLI interface
* Virtual environments
* Temporary/intermediate artifact management
* GPU-aware inference

---

# 🛠️ Installation

## 1. Clone the repository

```bash
git clone <https://github.com/ujala100/PolyDub>
cd PolyDub
```

## 2. Create a virtual environment

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 4. Install FFmpeg

### macOS

```bash
brew install ffmpeg
```

### Ubuntu / Debian

```bash
sudo apt install ffmpeg
```

### Windows

Download FFmpeg and add its `bin` directory to your system `PATH`.

Verify:

```bash
ffmpeg -version
```

---

# ▶️ Usage

Basic:

```bash
python main.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

Or run interactively:

```bash
python main.py
```

The program will prompt for the YouTube URL.

---

# ⚡ Configuration

### Choose Whisper model

```bash
python main.py "<URL>" --model medium
```

Larger models generally provide improved transcription quality at the cost of additional compute and processing time.

---

### Select TTS voice

```bash
python main.py "<URL>" --voice en-US-JennyNeural
```

List available voices:

```bash
edge-tts --list-voices
```

---

### Use GPU

```bash
python main.py "<URL>" --device cuda
```

---

### Custom output

```bash
python main.py "<URL>" --out my_dub.mp4
```

---

### Preserve intermediate files

Useful for debugging individual pipeline stages:

```bash
python main.py "<URL>" --keep-work-dir
```

---

# 📊 Processing Flow

For every input video, the system performs:

```text
01  Download
        ↓
02  Extract audio
        ↓
03  Detect spoken language
        ↓
04  Transcribe speech
        ↓
05  Preserve timestamps
        ↓
06  Translate segments
        ↓
07  Generate English speech
        ↓
08  Measure generated duration
        ↓
09  Time-fit speech
        ↓
10  Place clips on timeline
        ↓
11  Mix audio
        ↓
12  Replace original audio
        ↓
13  Preserve original video stream
        ↓
14  Export .mp4
```

---

# 🧪 Evaluation

The system is designed to be evaluated on both **shorter and long-form content**.

### Required evaluation

| Test   | Input            |
| ------ | ---------------- |
| Test 1 | ~30-minute video |
| Test 2 | ~2-hour video    |

For each evaluation, record:

```text
Source video
Dubbed video
Total processing time
Whisper model
Device
Source language
Output language
```

---

# 📈 Evaluation Dimensions

The project is evaluated across three major technical dimensions.

## 1. Translation Quality

Questions:

* Is the original meaning preserved?
* Is the English natural?
* Are idioms translated appropriately?
* Are technical terms preserved?
* Does the translation remain coherent across segments?

---

## 2. Speech Quality

Questions:

* Does the synthesized voice sound natural?
* Is pronunciation understandable?
* Is speech intelligible at normal playback speed?
* Does time stretching introduce noticeable artifacts?

---

## 3. Temporal Alignment

Questions:

* Does English speech begin near the original segment?
* Does speech remain synchronized throughout the video?
* Does the dub drift over time?
* Are long translated segments handled gracefully?

---

# ⏱️ Why Segment-Level Processing?

A naïve implementation could:

```text
Video
 ↓
Entire transcript
 ↓
Translate entire transcript
 ↓
Generate one large audio file
 ↓
Replace original audio
```

This creates synchronization problems.

Instead, this project works at the segment level:

```text
Segment 1 → Translate → TTS → Fit → Place
Segment 2 → Translate → TTS → Fit → Place
Segment 3 → Translate → TTS → Fit → Place
...
```

This prevents small timing errors from accumulating into significant drift.

---

# 🧠 Engineering Trade-offs

## Translation

### Current approach

Free translation endpoint.

### Advantages

* No API key
* Simple integration
* Low setup complexity
* Easy experimentation

### Limitations

* Rate limits
* Internet dependency
* Less control over domain-specific terminology
* Translation quality depends on the external service

### Upgrade path

```text
deep-translator
      ↓
IndicTrans2 / NLLB
      ↓
Domain-specific translation
      ↓
LLM-assisted translation
```

---

## TTS

### Current approach

Edge TTS.

### Advantages

* Natural-sounding voices
* Easy integration
* Multiple voices
* No paid API requirement


# 🔮 Future Roadmap

The architecture intentionally leaves room for significant upgrades.

## Phase 1 — Core Pipeline

* [x] YouTube ingestion
* [x] Multilingual transcription
* [x] Timestamp extraction
* [x] English translation
* [x] TTS synthesis
* [x] Time fitting
* [x] Audio remixing
* [x] Video muxing

---

## Phase 2 — Multi-Speaker Dubbing

```text
Audio
 ↓
Whisper
 ↓
Speaker Diarization
 ↓
Speaker A ──→ English Voice A
Speaker B ──→ English Voice B
Speaker C ──→ English Voice C
```

Potential technologies:

* pyannote.audio
* speaker embeddings
* XTTS
* voice cloning models

---

## Phase 3 — Better Translation

Replace the generic translator with:

* IndicTrans2
* NLLB
* domain-specific translation models
* LLM-based contextual translation

Potential improvement:

Instead of translating each segment independently:

```text
Segment N
```

provide contextual information:

```text
Previous segments
       +
Current segment
       +
Following context
       ↓
Context-aware translation
```

This can improve consistency for pronouns, terminology and conversational context.

---

## Phase 4 — Voice Preservation

Future versions can attempt:

```text
Original Speaker
       ↓
Speaker embedding
       ↓
Voice cloning
       ↓
English speech
```

The objective becomes:

> **Preserve not only what the speaker says, but also aspects of who is saying it.**

---

## Phase 5 — Audio Separation

Instead of simply replacing the entire soundtrack:

```text
Original audio
      ↓
 ┌────┴─────┐
 ↓          ↓
Speech     Background
 ↓          ↓
Translate   Preserve
 ↓          ↓
English     Music/SFX
 ↓          ↓
 └────┬─────┘
      ↓
   Final Mix
```

This would produce a more professional localization experience.

---

## Phase 6 — Production Architecture

Potential evolution:

```text
                    ┌───────────────┐
                    │  Web / API    │
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │ Job Queue     │
                    └───────┬───────┘
                            ↓
                 ┌──────────┴──────────┐
                 ↓                     ↓
          Transcription Worker   Translation Worker
                 ↓                     ↓
                 └──────────┬──────────┘
                            ↓
                     TTS Worker
                            ↓
                     Audio Worker
                            ↓
                     Video Worker
                            ↓
                    Object Storage
```

Potential technologies:

* FastAPI
* Celery / Redis
* Docker
* PostgreSQL
* object storage
* GPU workers
* monitoring
* distributed job processing

---

# 🔐 Responsible Use

This system is intended for legitimate video localization and accessibility use cases.

Potential applications include:

* Educational content localization
* Multilingual YouTube content
* Accessibility
* International audiences
* Training material localization
* Internal corporate videos
* Language learning
* Research and experimentation

When using voice cloning or publishing dubbed content, appropriate permission and disclosure should be obtained where required.

---

# 🎓 What This Project Demonstrates

This project goes beyond calling an AI API.

It demonstrates practical experience with:

### AI / ML

* Automatic Speech Recognition
* Multilingual NLP
* Machine Translation
* Text-to-Speech
* Speech processing
* Timestamp-aware inference
* Model selection
* AI pipeline design

### Software Engineering

* Modular architecture
* Component isolation
* CLI design
* Interface-driven pipeline stages
* Error handling
* Intermediate artifact management
* Dependency management

### Multimedia Engineering

* FFmpeg
* Audio extraction
* Audio time stretching
* Timeline composition
* Audio/video muxing
* Codec-aware processing

### System Design

* Replaceable model components
* Pipeline orchestration
* Long-running workloads
* GPU-aware processing
* Scalability considerations
* Quality/latency trade-offs

---

# 💡 Key Design Principle

> **Every AI component should be replaceable without rewriting the entire system.**

For example:

```text
Today

Whisper
   ↓
Google Translate
   ↓
Edge TTS
```

can become:

```text
Future

WhisperX
   ↓
IndicTrans2
   ↓
XTTS
```

without changing the overall architecture.

This separation makes experimentation, benchmarking and future productionization significantly easier.

---

# 📦 Output

The default output is written to:

```text
output/
└── dubbed_video.mp4
```

The final file contains:

```text
Original Video
        +
English Dubbed Audio
        ↓
Final MP4
```

The visual stream remains unchanged.

---

# ⚡ Performance Notes

Processing time depends heavily on:

* video duration
* Whisper model size
* CPU/GPU
* number of speech segments
* TTS latency
* translation latency
* internet speed
* FFmpeg processing

For reproducible benchmarking, always report:

```text
Video Duration
Model
Hardware
Device
Processing Time
Real-Time Factor
```

### Real-Time Factor

A useful metric is:

```text
RTF = Processing Time / Video Duration
```

For example:

```text
Video duration = 30 minutes
Processing time = 60 minutes

RTF = 60 / 30 = 2.0
```

Lower RTF means faster processing relative to the video's duration.

---

# 🧪 Example Benchmark Table

Populate this table with your **actual measured results** rather than estimated numbers.

| Test    | Duration | Language     | Whisper Model | Device      | Processing Time |     RTF |
| ------- | -------: | ------------ | ------------- | ----------- | --------------: | ------: |
| Video 1 |  ~30 min | `<language>` | `<model>`     | `<CPU/GPU>` |        `<time>` | `<RTF>` |
| Video 2 |    ~2 hr | `<language>` | `<model>`     | `<CPU/GPU>` |        `<time>` | `<RTF>` |

---

# 🏆 Assignment Requirements

The implementation addresses the core pipeline requirements:

| Requirement             | Implementation        |
| ----------------------- | --------------------- |
| Fetch video             | `yt-dlp`              |
| Transcribe              | `faster-whisper`      |
| Preserve timestamps     | Whisper segments      |
| Translate               | `deep-translator`     |
| Natural English voice   | `edge-tts`            |
| Timing alignment        | FFmpeg `atempo`       |
| Audio remix             | `pydub`               |
| Replace video audio     | FFmpeg                |
| Preserve visual quality | `-c:v copy`           |
| 30-minute evaluation    | Supported             |
| 2-hour evaluation       | Supported             |
| Multi-speaker dubbing   | Future / stretch goal |

---

# 👨‍💻 Running the Full Pipeline

```bash
python main.py \
  "https://www.youtube.com/watch?v=XXXXXXXXXXX" \
  --model medium \
  --voice en-US-JennyNeural \
  --device cuda \
  --out output/dubbed_video.mp4
```

For debugging:

```bash
python main.py "<URL>" --keep-work-dir
```

---

# 🗺️ High-Level System Design

```text
                         USER
                          │
                          ▼
                    YouTube URL
                          │
                          ▼
                  ┌───────────────┐
                  │   yt-dlp      │
                  └───────┬───────┘
                          │
                       Video
                          │
                          ▼
                  ┌───────────────┐
                  │    FFmpeg     │
                  │ Audio Extract │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ faster-whisper│
                  └───────┬───────┘
                          │
                 Timestamped Segments
                          │
                          ▼
                  ┌───────────────┐
                  │  Translator   │
                  └───────┬───────┘
                          │
                     English Text
                          │
                          ▼
                  ┌───────────────┐
                  │    Edge TTS   │
                  └───────┬───────┘
                          │
                   English Audio
                          │
                          ▼
                  ┌───────────────┐
                  │ FFmpeg atempo │
                  │ Time Fitting  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │    pydub      │
                  │ Timeline Mix  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │    FFmpeg     │
                  │   -c:v copy   │
                  └───────┬───────┘
                          │
                          ▼
                 🎬 Dubbed MP4
```

---

# 📌 Project Status

**Core pipeline:** ✅ Complete

**Multilingual transcription:** ✅

**English translation:** ✅

**Natural TTS:** ✅

**Timestamp synchronization:** ✅

**Video-preserving mux:** ✅

**Multi-speaker diarization:** 🚧 Planned

**Voice cloning:** 🚧 Planned

**Background audio preservation:** 🚧 Planned

**Context-aware translation:** 🚧 Planned

**Production API:** 🚧 Planned

---

# 📜 License

Add the license appropriate for your repository and dependencies.

---

# ⭐ If You Find This Project Interesting

This project explores the intersection of:

**Speech AI × NLP × Generative AI × Multimedia Processing × System Design**

If you are interested in multilingual AI, speech technology, video localization, or AI infrastructure, feel free to explore the architecture and implementation.

---

## Built With

```text
Python
faster-whisper
FFmpeg
yt-dlp
deep-translator
Edge TTS
pydub
```

**From a YouTube URL to an English-dubbed video — through a modular, timestamp-aware AI pipeline.**
