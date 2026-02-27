"""FastAPI web server for voice-analyzer."""

import asyncio
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Ensure voice_analyzer package is importable when run from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_analyzer.audio import load_audio
from voice_analyzer.diarize import (
    align_transcript_to_speakers,
    diarize_speakers,
    identify_speaker_roles,
    verify_speaker_transcripts,
)
from voice_analyzer.enhance import enhance_audio
from voice_analyzer.medlatec import generate_call_summary
from voice_analyzer.output import build_conversation_timeline, build_json_output
from voice_analyzer.pitch import analyze_pitch, plot_pitch_contour
from voice_analyzer.transcribe import transcribe, transcribe_with_timestamps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("voice_analyzer.web")

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "voice_analyzer_web"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Voice Analyzer")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Per-job stores
_job_queues: dict[str, queue.Queue] = {}
_job_pitch_files: dict[str, str] = {}


# ── Analysis worker ────────────────────────────────────────────────────────────

def _push(q: queue.Queue, step: int, status: str, message: str, detail: str = "") -> None:
    q.put({"type": "progress", "step": step, "status": status,
           "message": message, "detail": detail})


def _run_analysis(job_id: str, file_path: str, options: dict) -> None:
    q = _job_queues[job_id]
    pitch_png = str(UPLOAD_DIR / f"{job_id}_pitch.png")

    try:
        model = options.get("model_size", "large")
        device = options.get("device", "auto")
        skip_diar = options.get("skip_diarization", False)

        # ── Step 1: Load & enhance audio ──────────────────────────────────────
        _push(q, 1, "running", "Loading audio…")
        audio, sr = load_audio(file_path)
        duration = len(audio) / sr
        _push(q, 1, "running", "Enhancing audio (noise reduction)…", f"{duration:.1f}s · {sr} Hz")
        audio = enhance_audio(audio, sr)
        _push(q, 1, "done", "Audio loaded & enhanced", f"{duration:.1f}s · {sr} Hz")

        # ── Step 2: Transcribe ─────────────────────────────────────────────────
        _push(q, 2, "running", f"Transcribing speech (PhoWhisper-{model})…")
        if skip_diar:
            transcript = transcribe(audio, sr, model_size=model, device=device)
            timed_chunks = []
        else:
            timed_chunks = transcribe_with_timestamps(audio, sr, model_size=model, device=device)
            # Word-level chunks: preserve spaces already embedded in tokens
            transcript = "".join(
                c["text"] if c["text"].startswith(" ") else " " + c["text"]
                for c in timed_chunks
            ).strip()
        _push(q, 2, "done", "Transcription complete", f"{len(transcript)} characters")

        # ── Step 3: Pitch ──────────────────────────────────────────────────────
        _push(q, 3, "running", "Analysing pitch…")
        pitch_stats = analyze_pitch(audio, sr)
        plot_pitch_contour(pitch_stats, pitch_png)
        _job_pitch_files[job_id] = pitch_png
        _push(q, 3, "done", "Pitch analysed",
              f"Mean {pitch_stats['mean_hz']:.1f} Hz · {pitch_stats['classification'].upper()}")

        # ── Step 4: Diarization ────────────────────────────────────────────────
        speaker_segments: list = []
        aligned_transcripts: dict = {}   # {spk: [{text,start,end}]} — has timestamps
        speaker_transcripts: dict = {}   # {spk: [{text,start,end}]} — verified text
        roles: dict = {}
        conversation: list = []

        if skip_diar:
            _push(q, 4, "skipped", "Diarization skipped", "")
            _push(q, 5, "skipped", "Verification skipped", "")
        else:
            hf_token = options.get("hf_token") or os.environ.get("HF_TOKEN", "")
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
            _push(q, 4, "running", "Diarizing speakers (pyannote/speaker-diarization-3.1)…")
            try:
                speaker_segments = diarize_speakers(
                    audio, sr,
                    device=device,
                    num_speakers=options.get("num_speakers"),
                )
                # Aligned transcript keeps precise word-level timestamps
                aligned_transcripts = align_transcript_to_speakers(speaker_segments, timed_chunks)
                n = len({s["speaker"] for s in speaker_segments})
                _push(q, 4, "done", "Diarization complete",
                      f"{n} speaker{'s' if n != 1 else ''} · {len(speaker_segments)} segments")

                # ── Step 5: Verify + role identification ──────────────────────
                _push(q, 5, "running", "Verifying transcripts & identifying roles…",
                      "Re-transcribing each speaker's audio")
                verified = verify_speaker_transcripts(
                    audio, sr, speaker_segments,
                    model_size=model, device=device,
                )
                # Build speaker_transcripts for summary cards (verified text)
                for spk_id, text in verified.items():
                    speaker_transcripts[spk_id] = [{"text": text, "start": 0, "end": 0}] if text else []

                # Identify agent vs customer using aligned (timestamped) data
                roles = identify_speaker_roles(speaker_segments, aligned_transcripts)

                # Build conversation timeline BEFORE stripping f0 arrays
                conversation = build_conversation_timeline(aligned_transcripts, pitch_stats, roles)

                role_summary = ", ".join(f"{k}={v}" for k, v in roles.items())
                _push(q, 5, "done", "Verification complete",
                      f"{len(verified)} transcripts verified · {role_summary}")

            except Exception as exc:
                logger.warning("Diarization failed: %s", exc)
                _push(q, 4, "warning", "Diarization skipped", str(exc)[:120])
                _push(q, 5, "skipped", "Verification skipped", "")

        # ── Build result ───────────────────────────────────────────────────────
        result = build_json_output(
            transcript, pitch_stats, speaker_segments, speaker_transcripts,
            roles=roles, conversation=conversation,
        )
        result["duration"] = round(duration, 2)
        # Strip large f0 arrays — PNG is used for chart; window averages already computed
        result["pitch"].pop("f0_times", None)
        result["pitch"].pop("f0_values", None)

        # ── MEDLATEC call summary ──────────────────────────────────────────────
        if conversation:
            result["call_summary"] = generate_call_summary(
                conversation, transcript, roles, duration,
            )

        q.put({"type": "done", "results": result})

    except Exception as exc:
        logger.exception("Analysis failed for job %s", job_id)
        q.put({"type": "error", "message": str(exc)})
    finally:
        try:
            os.unlink(file_path)
        except Exception:
            pass


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    model_size: str = Form("large"),
    num_speakers: str = Form(""),
    device: str = Form("auto"),
    skip_diarization: str = Form("false"),
    hf_token: str = Form(""),
) -> dict:
    job_id = str(uuid.uuid4())
    suffix = Path(file.filename or "audio.mp3").suffix.lower() or ".mp3"
    tmp_path = str(UPLOAD_DIR / f"{job_id}{suffix}")

    with open(tmp_path, "wb") as fh:
        fh.write(await file.read())

    options = {
        "model_size": model_size,
        "num_speakers": int(num_speakers) if num_speakers.strip().isdigit() else None,
        "device": device,
        "skip_diarization": skip_diarization.lower() == "true",
        "hf_token": hf_token.strip() or None,
    }

    _job_queues[job_id] = queue.Queue()
    threading.Thread(
        target=_run_analysis, args=(job_id, tmp_path, options), daemon=True
    ).start()

    return {"job_id": job_id}


@app.get("/stream/{job_id}")
async def stream_events(job_id: str) -> StreamingResponse:
    async def generator():
        q = _job_queues.get(job_id)
        if not q:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
            return

        loop = asyncio.get_event_loop()
        while True:
            try:
                event = await loop.run_in_executor(None, lambda: q.get(timeout=300))
            except Exception:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Processing timeout'})}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] in ("done", "error"):
                _job_queues.pop(job_id, None)
                break

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/pitch/{job_id}")
async def serve_pitch(job_id: str):
    path = _job_pitch_files.get(job_id)
    if not path or not Path(path).exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=False)
