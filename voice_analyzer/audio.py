"""Audio loading and preprocessing utilities."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}
TARGET_SR = 16000  # 16kHz required by Whisper and pyannote


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file and resample to 16kHz mono.

    Args:
        file_path: Path to the audio file (WAV, MP3, M4A, etc.).

    Returns:
        Tuple of (audio array as float32, sample rate).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is not supported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    logger.info("Loading audio file: %s", file_path)

    # Use pydub to handle non-WAV formats, then hand off to librosa
    if suffix in {".mp3", ".m4a", ".aac", ".ogg"}:
        audio_path = _convert_to_wav(file_path)
        temp = True
    else:
        audio_path = file_path
        temp = False

    try:
        audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    finally:
        if temp and os.path.exists(audio_path):
            os.remove(audio_path)

    logger.info(
        "Loaded audio: %.1f seconds at %d Hz (%d samples)",
        len(audio) / sr,
        sr,
        len(audio),
    )
    return audio.astype(np.float32), sr


def _convert_to_wav(file_path: str) -> str:
    """Convert a non-WAV audio file to a temporary WAV file.

    Args:
        file_path: Source audio file path.

    Returns:
        Path to the temporary WAV file (caller must delete).
    """
    logger.info("Converting %s to temporary WAV for processing", file_path)
    segment = AudioSegment.from_file(file_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    segment.export(tmp.name, format="wav")
    tmp.close()
    return tmp.name


def save_wav(audio: np.ndarray, sr: int, file_path: str) -> None:
    """Save a numpy audio array as a WAV file.

    Args:
        audio: Audio samples as float32.
        sr: Sample rate.
        file_path: Destination path.
    """
    sf.write(file_path, audio, sr)
    logger.info("Saved WAV: %s", file_path)


def chunk_audio(audio: np.ndarray, sr: int, chunk_sec: int = 30) -> list:
    """Split audio into fixed-length chunks for long-file processing.

    Args:
        audio: Audio samples.
        sr: Sample rate.
        chunk_sec: Chunk length in seconds (default 30).

    Returns:
        List of (start_time_sec, chunk_array) tuples.
    """
    chunk_len = chunk_sec * sr
    chunks = []
    for i in range(0, len(audio), chunk_len):
        start_sec = i / sr
        chunks.append((start_sec, audio[i : i + chunk_len]))
    return chunks
