"""Voice pitch (F0) analysis using librosa."""

import logging
from typing import Optional, Tuple

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for PNG export

logger = logging.getLogger(__name__)

# Gender-based pitch classification thresholds (Hz)
FEMALE_HIGH_THRESHOLD = 200.0
MALE_HIGH_THRESHOLD = 150.0


def analyze_pitch(
    audio: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 600.0,
) -> dict:
    """Estimate fundamental frequency (F0) and compute pitch statistics.

    Uses librosa's pyin algorithm for robust pitch estimation on voiced frames.

    Args:
        audio: Audio samples as float32 at 16kHz.
        sr: Sample rate.
        fmin: Minimum expected pitch frequency in Hz.
        fmax: Maximum expected pitch frequency in Hz.

    Returns:
        Dict with keys:
            mean_hz, min_hz, max_hz, median_hz, std_hz,
            classification ("high"/"low"),
            f0_times (list[float]), f0_values (list[float|None])
    """
    logger.info("Analyzing pitch with pyin (fmin=%.0f Hz, fmax=%.0f Hz)", fmin, fmax)

    # pyin returns (f0, voiced_flag, voiced_probs)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
    )

    times = librosa.times_like(f0, sr=sr)

    # Only use voiced frames for statistics
    voiced_f0 = f0[voiced_flag]

    if len(voiced_f0) == 0:
        logger.warning("No voiced frames detected; pitch statistics will be zero.")
        stats = {
            "mean_hz": 0.0,
            "min_hz": 0.0,
            "max_hz": 0.0,
            "median_hz": 0.0,
            "std_hz": 0.0,
            "classification": "unknown",
        }
    else:
        mean_hz = float(np.mean(voiced_f0))
        stats = {
            "mean_hz": round(mean_hz, 2),
            "min_hz": round(float(np.min(voiced_f0)), 2),
            "max_hz": round(float(np.max(voiced_f0)), 2),
            "median_hz": round(float(np.median(voiced_f0)), 2),
            "std_hz": round(float(np.std(voiced_f0)), 2),
            "classification": _classify_pitch(mean_hz),
        }

    # Store full contour (None where unvoiced) for plotting
    f0_values = [
        round(float(v), 2) if voiced_flag[i] else None for i, v in enumerate(f0)
    ]
    stats["f0_times"] = [round(float(t), 4) for t in times]
    stats["f0_values"] = f0_values

    logger.info(
        "Pitch stats: mean=%.1f Hz, min=%.1f Hz, max=%.1f Hz, class=%s",
        stats["mean_hz"],
        stats["min_hz"],
        stats["max_hz"],
        stats["classification"],
    )
    return stats


def _classify_pitch(mean_hz: float) -> str:
    """Classify pitch as 'high' or 'low' based on mean F0.

    Uses conservative thresholds:
      - high: mean > 200 Hz  (typical female range)
      - low:  mean <= 200 Hz (typical male range)

    Args:
        mean_hz: Mean fundamental frequency.

    Returns:
        "high" or "low".
    """
    if mean_hz >= FEMALE_HIGH_THRESHOLD:
        return "high"
    elif mean_hz >= MALE_HIGH_THRESHOLD:
        return "mid"
    else:
        return "low"


def plot_pitch_contour(
    pitch_stats: dict,
    output_path: str,
    title: str = "Pitch Contour",
) -> None:
    """Generate and save a pitch contour plot as a PNG file.

    Args:
        pitch_stats: Output dict from analyze_pitch().
        output_path: Destination PNG file path.
        title: Plot title.
    """
    times = pitch_stats["f0_times"]
    f0_values = pitch_stats["f0_values"]

    # Separate voiced frames for scatter plot
    voiced_times = [t for t, v in zip(times, f0_values) if v is not None]
    voiced_vals = [v for v in f0_values if v is not None]

    fig, ax = plt.subplots(figsize=(12, 4))

    if voiced_times:
        ax.scatter(voiced_times, voiced_vals, s=2, color="#2196F3", alpha=0.7, label="F0")
        ax.axhline(
            pitch_stats["mean_hz"],
            color="#F44336",
            linestyle="--",
            linewidth=1.2,
            label=f"Mean: {pitch_stats['mean_hz']:.1f} Hz",
        )
        ax.set_ylim(
            max(0, pitch_stats["min_hz"] - 20),
            pitch_stats["max_hz"] + 40,
        )

    # Reference lines
    ax.axhline(FEMALE_HIGH_THRESHOLD, color="#9C27B0", linestyle=":", linewidth=0.8, alpha=0.6,
               label=f"Female threshold ({FEMALE_HIGH_THRESHOLD:.0f} Hz)")
    ax.axhline(MALE_HIGH_THRESHOLD, color="#4CAF50", linestyle=":", linewidth=0.8, alpha=0.6,
               label=f"Male threshold ({MALE_HIGH_THRESHOLD:.0f} Hz)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotation box
    stats_text = (
        f"Mean: {pitch_stats['mean_hz']:.1f} Hz\n"
        f"Min:  {pitch_stats['min_hz']:.1f} Hz\n"
        f"Max:  {pitch_stats['max_hz']:.1f} Hz\n"
        f"Class: {pitch_stats['classification'].upper()}"
    )
    ax.text(
        0.01, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Pitch contour plot saved: %s", output_path)
