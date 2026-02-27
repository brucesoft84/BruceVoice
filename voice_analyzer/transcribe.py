"""Vietnamese speech-to-text transcription using PhoWhisper (vinai/PhoWhisper-*)."""

import logging
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .audio import TARGET_SR, chunk_audio

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "tiny": "vinai/PhoWhisper-tiny",
    "base": "vinai/PhoWhisper-base",
    "small": "vinai/PhoWhisper-small",
    "medium": "vinai/PhoWhisper-medium",
    "large": "vinai/PhoWhisper-large",
}

_pipeline_cache: dict = {}


def _get_device(device: str) -> str:
    """Resolve device string to 'cuda' or 'cpu'."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model(model_size: str = "base", device: str = "auto") -> pipeline:
    """Load (or return cached) PhoWhisper pipeline.

    Args:
        model_size: One of tiny/base/small/medium/large.
        device: 'auto', 'cpu', or 'cuda'.

    Returns:
        HuggingFace ASR pipeline.
    """
    resolved_device = _get_device(device)
    cache_key = (model_size, resolved_device)

    if cache_key in _pipeline_cache:
        logger.info("Using cached PhoWhisper model (%s)", model_size)
        return _pipeline_cache[cache_key]

    model_id = MODEL_MAP.get(model_size)
    if model_id is None:
        raise ValueError(f"Unknown model size '{model_size}'. Choose from: {list(MODEL_MAP)}")

    logger.info("Loading PhoWhisper model: %s on %s", model_id, resolved_device)

    torch_dtype = torch.float16 if resolved_device == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(resolved_device)

    processor = AutoProcessor.from_pretrained(model_id)

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=resolved_device,
        return_timestamps=True,
    )

    _pipeline_cache[cache_key] = asr_pipeline
    return asr_pipeline


def transcribe(
    audio: np.ndarray,
    sr: int,
    model_size: str = "base",
    device: str = "auto",
    chunk_length_s: int = 30,
) -> str:
    """Transcribe audio to Vietnamese text using PhoWhisper.

    Long audio (>30s) is automatically chunked and the results concatenated.

    Args:
        audio: Audio samples as float32 at 16kHz.
        sr: Sample rate (should be TARGET_SR = 16000).
        model_size: PhoWhisper model size.
        device: 'auto', 'cpu', or 'cuda'.
        chunk_length_s: Max chunk length in seconds for batched inference.

    Returns:
        Full transcript string.
    """
    asr = load_model(model_size, device)
    duration = len(audio) / sr

    logger.info("Transcribing %.1f seconds of audio with PhoWhisper-%s", duration, model_size)

    with torch.no_grad():
        if duration <= chunk_length_s:
            result = asr(
                {"array": audio, "sampling_rate": sr},
                generate_kwargs={"language": "vi", "task": "transcribe"},
            )
            transcript = result["text"].strip()
        else:
            # Process in chunks with progress bar
            chunks = chunk_audio(audio, sr, chunk_length_s)
            parts = []
            for start_sec, chunk in tqdm(chunks, desc="Transcribing chunks", unit="chunk"):
                res = asr(
                    {"array": chunk, "sampling_rate": sr},
                    generate_kwargs={"language": "vi", "task": "transcribe"},
                )
                parts.append(res["text"].strip())
            transcript = " ".join(parts)

    logger.info("Transcription complete: %d characters", len(transcript))
    return transcript


def transcribe_with_timestamps(
    audio: np.ndarray,
    sr: int,
    model_size: str = "base",
    device: str = "auto",
    chunk_length_s: int = 30,
    word_timestamps: bool = True,
) -> list:
    """Transcribe audio and return word-level (or segment-level) timestamps.

    Word-level timestamps (default) give much finer granularity (~0.1–0.3 s per
    token), which dramatically improves speaker-transcript alignment when
    combined with diarization.

    Args:
        audio: Audio samples as float32.
        sr: Sample rate.
        model_size: PhoWhisper model size.
        device: Compute device.
        chunk_length_s: Chunk size for long audio.
        word_timestamps: When True (default) each returned chunk is a single
            word/token; when False chunks are Whisper decoder segments.

    Returns:
        List of dicts: [{"text": str, "timestamp": (start, end)}, ...]
    """
    asr = load_model(model_size, device)
    duration = len(audio) / sr

    logger.info(
        "Transcribing with %s timestamps (%.1f s)",
        "word" if word_timestamps else "segment",
        duration,
    )

    with torch.no_grad():
        result = asr(
            {"array": audio, "sampling_rate": sr},
            generate_kwargs={
                "language": "vi",
                "task": "transcribe",
                "num_beams": 5,
            },
            chunk_length_s=chunk_length_s,
            return_timestamps="word" if word_timestamps else True,
        )

    chunks = result.get("chunks", [])
    segments = []
    for chunk in chunks:
        text = chunk["text"].strip()
        if not text:
            continue
        segments.append(
            {
                "text": text,
                "timestamp": chunk["timestamp"],  # (start_sec, end_sec)
            }
        )
    return segments
