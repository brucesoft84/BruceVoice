"""Audio quality enhancement: noise reduction and amplitude normalization."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def denoise_audio(audio: np.ndarray, sr: int, prop_decrease: float = 0.80) -> np.ndarray:
    """Apply spectral-gating noise reduction via noisereduce.

    Uses a non-stationary (adaptive) algorithm that estimates the noise floor
    across the full signal and attenuates it without requiring a separate
    noise-only sample.

    Args:
        audio: Audio samples as float32 at any sample rate.
        sr: Sample rate in Hz.
        prop_decrease: Fraction of noise to remove (0–1). 0.80 removes 80 %
            of the estimated noise energy while keeping speech intact.

    Returns:
        Denoised audio as float32. Falls back to the original array if
        noisereduce is not installed.
    """
    try:
        import noisereduce as nr
    except ImportError:
        logger.warning(
            "noisereduce not installed — skipping denoising. "
            "Install with: pip install noisereduce"
        )
        return audio

    denoised = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,      # adaptive: works for phone/room noise
        prop_decrease=prop_decrease,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
    )
    return denoised.astype(np.float32)


def normalize_audio(audio: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    """Normalize audio to a consistent RMS amplitude.

    Brings quiet recordings up and loud ones down so both the transcription
    model and the diarization pipeline receive a well-levelled signal.

    Args:
        audio: Audio samples as float32.
        target_rms: Desired RMS level (default 0.08 ≈ −22 dBFS).

    Returns:
        Amplitude-normalized audio as float32.
    """
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 1e-9:
        return audio  # silent — nothing to do
    gain = target_rms / rms
    # Cap gain to avoid boosting very quiet (likely noise-only) passages
    gain = min(gain, 10.0)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def enhance_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Full pre-processing pipeline: denoise → normalize.

    Apply this to the raw loaded audio before passing it to the
    transcription and diarization pipelines.

    Args:
        audio: Raw audio samples as float32.
        sr: Sample rate in Hz.

    Returns:
        Enhanced audio ready for downstream analysis.
    """
    logger.info("Enhancing audio: spectral noise reduction + RMS normalization")
    audio = denoise_audio(audio, sr)
    audio = normalize_audio(audio)
    return audio
