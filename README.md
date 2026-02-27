# voice-analyzer

A production-ready CLI tool for Vietnamese audio analysis:

- **Speech-to-Text** – Vietnamese transcription via [PhoWhisper](https://huggingface.co/vinai/PhoWhisper-large) (VinAI)
- **Pitch Analysis** – Fundamental frequency (F0) statistics and contour plot via librosa
- **Speaker Diarization** – Who spoke when, with per-speaker transcripts via pyannote.audio

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) (for MP3/M4A support via pydub)
- A [Hugging Face](https://huggingface.co/) account with access to:
  - `vinai/PhoWhisper-*` (accept model terms on HF Hub)
  - `pyannote/speaker-diarization-3.1` (accept model terms on HF Hub)

---

## Installation

```bash
# Clone / download the project
cd voice-analyzer

# Install dependencies
pip install -r requirements.txt

# Install the CLI
pip install -e .
```

### Hugging Face token

Diarization requires a HF token:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## Usage

```
voice-analyzer [OPTIONS] AUDIO_FILE
```

### Options

| Option | Default | Description |
|---|---|---|
| `--output-text FILE` | stdout | Save transcript to a text file |
| `--output-json FILE` | – | Save full JSON report |
| `--output-pitch FILE` | – | Save pitch contour PNG |
| `--model-size` | `base` | PhoWhisper size: tiny/base/small/medium/large |
| `--num-speakers INT` | auto | Expected speaker count for diarization |
| `--device` | `auto` | Compute device: auto/cpu/cuda |
| `--skip-diarization` | off | Skip diarization (faster, no HF token needed) |
| `--help` | | Show help |

### Examples

```bash
# Quick transcription (stdout)
voice-analyzer recording.wav --skip-diarization

# Full analysis with all outputs
voice-analyzer interview.mp3 \
  --output-text transcript.txt \
  --output-json report.json \
  --output-pitch pitch.png \
  --model-size large \
  --num-speakers 2

# CPU-only, medium model
voice-analyzer call.m4a --model-size medium --device cpu
```

---

## Output Format

### Console

Structured, colored output showing transcript, pitch summary, speaker timeline, and per-speaker transcripts.

### JSON (`--output-json`)

```json
{
  "transcript": "...",
  "pitch": {
    "mean_hz": 195.3,
    "min_hz": 82.4,
    "max_hz": 440.0,
    "median_hz": 190.1,
    "std_hz": 42.7,
    "classification": "mid"
  },
  "speakers": [
    {
      "id": "SPEAKER_00",
      "segments": [{"start": 0.5, "end": 12.3}],
      "transcript": "Xin chào..."
    }
  ]
}
```

### Pitch classification

| Classification | Mean F0 |
|---|---|
| `high` | ≥ 200 Hz (typical female) |
| `mid` | 150–199 Hz |
| `low` | < 150 Hz (typical male) |

---

## Running Tests

```bash
pytest -v
```

---

## Project Structure

```
voice-analyzer/
├── voice_analyzer/
│   ├── __init__.py
│   ├── main.py        # CLI entry point (Click)
│   ├── audio.py       # load_audio(), chunk_audio()
│   ├── transcribe.py  # transcribe() via PhoWhisper
│   ├── pitch.py       # analyze_pitch(), plot_pitch_contour()
│   ├── diarize.py     # diarize_speakers(), align_transcript_to_speakers()
│   └── output.py      # Console/JSON/text output helpers
├── tests/
│   ├── test_audio.py
│   ├── test_pitch.py
│   └── test_transcribe.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Notes

- Models are **auto-downloaded** on first run and cached by HuggingFace/torch.
- Long audio files (>30 s) are automatically chunked for transcription.
- The `--skip-diarization` flag is useful for quick transcription-only workflows.
- GPU (CUDA) is used automatically when available; set `--device cpu` to force CPU.
