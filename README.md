# BruceVoice — Vietnamese Voice Analyzer

A production-ready tool for Vietnamese audio analysis with a real-time web UI and CLI.

| Feature | Technology |
|---|---|
| Speech-to-Text | [PhoWhisper](https://huggingface.co/vinai/PhoWhisper-large) (VinAI) — word-level timestamps, beam search |
| Speaker Diarization | [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) |
| Transcript Verification | Per-speaker re-transcription for cross-validation |
| Pitch Analysis | librosa pyin — F0 statistics + contour plot |
| Audio Enhancement | Spectral noise reduction (noisereduce) + RMS normalization |
| Web UI | FastAPI + SSE real-time progress + single-page Tailwind frontend |

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) — for MP3/M4A decoding
- A [Hugging Face](https://huggingface.co/) account with access to:
  - [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) — accept terms on HF Hub
  - [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) — accept terms on HF Hub

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/brucesoft84/BruceVoice.git
cd BruceVoice

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the CLI tool
pip install -e .
```

### ffmpeg (macOS)

```bash
brew install ffmpeg
```

### Hugging Face Token

Diarization requires a token with access to the pyannote models. Generate one at https://huggingface.co/settings/tokens then export it:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## Web UI

The recommended way to use BruceVoice is through the browser interface.

### Start the server

```bash
HF_TOKEN=hf_xxx venv/bin/uvicorn web.app:app \
  --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

### 5-step real-time pipeline

Progress is streamed live to the browser via Server-Sent Events:

| Step | Description |
|---|---|
| 1 — Load & Enhance | Load audio, apply spectral noise reduction and RMS normalization |
| 2 — Transcribe | PhoWhisper-large with word-level timestamps and beam search decoding |
| 3 — Pitch | F0 estimation, statistics, pitch contour PNG |
| 4 — Diarize | Speaker segmentation, word-to-speaker alignment |
| 5 — Verify | Re-transcribe each speaker's audio independently for accuracy |

### UI options

| Option | Description |
|---|---|
| Model | PhoWhisper size: tiny / base / small / medium / large |
| Speakers | Expected speaker count (leave blank for auto-detect) |
| Device | auto / cpu / cuda |
| Skip Diarization | Transcription-only mode (no HF token needed) |
| HF Token | Hugging Face access token (stored only in memory for the request) |

### Supported audio formats

WAV · MP3 · M4A · FLAC · OGG · AAC

---

## Expose to the Internet

### Tailscale Funnel (stable URL, recommended)

```bash
# Install Tailscale from https://tailscale.com/download or Mac App Store
tailscale funnel 8000
# → https://macbook-pro-ca-admin.tail425a25.ts.net/
```

### Cloudflare Quick Tunnel (no account needed)

```bash
brew install cloudflare/cloudflare/cloudflared
cloudflared tunnel --url http://localhost:8000
# → https://<random>.trycloudflare.com
```

---

## CLI

```bash
voice-analyzer [OPTIONS] AUDIO_FILE
```

### Options

| Option | Default | Description |
|---|---|---|
| `--output-text FILE` | stdout | Save transcript to a text file |
| `--output-json FILE` | — | Save full JSON report |
| `--output-pitch FILE` | — | Save pitch contour PNG |
| `--model-size` | `base` | PhoWhisper size: tiny / base / small / medium / large |
| `--num-speakers INT` | auto | Expected speaker count for diarization |
| `--device` | `auto` | Compute device: auto / cpu / cuda |
| `--skip-diarization` | off | Skip diarization (faster, no HF token needed) |

### Examples

```bash
# Quick transcription only
voice-analyzer recording.wav --skip-diarization

# Full analysis — 2 speakers, large model, all outputs saved
voice-analyzer interview.mp3 \
  --output-text transcript.txt \
  --output-json report.json \
  --output-pitch pitch.png \
  --model-size large \
  --num-speakers 2

# CPU-only (no GPU)
voice-analyzer call.m4a --model-size medium --device cpu
```

---

## Output Format

### Console

Structured, colored output with transcript, pitch summary, speaker timeline, and per-speaker transcripts.

### JSON (`--output-json`)

```json
{
  "transcript": "xin chào em ơi...",
  "duration": 26.81,
  "pitch": {
    "mean_hz": 191.6,
    "min_hz": 50.0,
    "max_hz": 592.4,
    "median_hz": 183.4,
    "std_hz": 80.3,
    "classification": "mid"
  },
  "speakers": [
    {
      "id": "SPEAKER_00",
      "segments": [
        { "start": 0.4, "end": 2.5 },
        { "start": 11.0, "end": 18.2 }
      ],
      "transcript": "xin chào em ơi cho anh hỏi..."
    },
    {
      "id": "SPEAKER_01",
      "segments": [
        { "start": 3.3, "end": 10.1 }
      ],
      "transcript": "dạ anh đang ở khu vực nào ạ..."
    }
  ]
}
```

### Pitch classification

| Classification | Mean F0 | Typical voice |
|---|---|---|
| `high` | ≥ 200 Hz | Female |
| `mid` | 150–199 Hz | Mixed / varies |
| `low` | < 150 Hz | Male |

---

## Project Structure

```
BruceVoice/
├── voice_analyzer/
│   ├── __init__.py
│   ├── main.py         # CLI entry point (Click)
│   ├── audio.py        # load_audio(), chunk_audio()
│   ├── enhance.py      # enhance_audio(): denoise + normalize
│   ├── transcribe.py   # transcribe(), transcribe_with_timestamps() via PhoWhisper
│   ├── pitch.py        # analyze_pitch(), plot_pitch_contour()
│   ├── diarize.py      # diarize_speakers(), align_transcript_to_speakers(),
│   │                   # verify_speaker_transcripts()
│   └── output.py       # Console / JSON / text output helpers
├── web/
│   ├── __init__.py
│   ├── app.py          # FastAPI server — SSE streaming, 5 endpoints
│   └── static/
│       └── index.html  # Single-page UI (Tailwind CDN, vanilla JS)
├── tests/
│   ├── test_audio.py
│   ├── test_pitch.py
│   └── test_transcribe.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Running Tests

```bash
pytest -v
```

---

## Troubleshooting

### `list_audio_backends` AttributeError
torchaudio 2.4+ removed this function. The patch in `diarize.py` handles it automatically.

### pyannote 403 on community-1 model
`speaker-diarization-3.1` uses `AgglomerativeClustering` which does not need the PLDA weights. The `get_plda` patch in `diarize.py` suppresses this silently.

### Empty speaker transcripts
Ensure `num_speakers` matches the actual number of speakers in the recording. With auto-detect, very short recordings may confuse the model.

### Models not downloading
Make sure your HF token has accepted the terms for both `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on the Hugging Face website.

---

## Notes

- Models are **auto-downloaded** on first run and cached by HuggingFace.
- Long audio (> 30 s) is automatically chunked for transcription.
- GPU (CUDA) is used automatically when available; set `--device cpu` to force CPU.
- Audio is enhanced (denoised + normalized) before all analysis steps.
- The verification step re-transcribes each speaker's audio in isolation, which significantly improves per-speaker accuracy.
