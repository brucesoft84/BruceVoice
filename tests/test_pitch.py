"""Unit tests for pitch analysis utilities."""

import os
import tempfile

import numpy as np
import pytest

from voice_analyzer.audio import TARGET_SR
from voice_analyzer.pitch import (
    FEMALE_HIGH_THRESHOLD,
    MALE_HIGH_THRESHOLD,
    _classify_pitch,
    analyze_pitch,
    plot_pitch_contour,
)


def _sine_wave(freq_hz: float, duration_s: float = 1.0, sr: int = TARGET_SR) -> np.ndarray:
    """Generate a pure sine wave at the given frequency."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


class TestClassifyPitch:
    def test_high_pitch(self):
        assert _classify_pitch(FEMALE_HIGH_THRESHOLD + 10) == "high"

    def test_mid_pitch(self):
        mid = (FEMALE_HIGH_THRESHOLD + MALE_HIGH_THRESHOLD) / 2
        assert _classify_pitch(mid) == "mid"

    def test_low_pitch(self):
        assert _classify_pitch(MALE_HIGH_THRESHOLD - 10) == "low"

    def test_boundary_female(self):
        assert _classify_pitch(FEMALE_HIGH_THRESHOLD) == "high"

    def test_boundary_male(self):
        assert _classify_pitch(MALE_HIGH_THRESHOLD) == "mid"


class TestAnalyzePitch:
    def test_returns_required_keys(self):
        audio = _sine_wave(220.0)
        stats = analyze_pitch(audio, TARGET_SR)
        required = {"mean_hz", "min_hz", "max_hz", "median_hz", "std_hz",
                    "classification", "f0_times", "f0_values"}
        assert required.issubset(stats.keys())

    def test_silent_audio_gives_zero_stats(self):
        audio = np.zeros(TARGET_SR, dtype=np.float32)
        stats = analyze_pitch(audio, TARGET_SR)
        assert stats["mean_hz"] == 0.0
        assert stats["classification"] == "unknown"

    def test_f0_lengths_match(self):
        audio = _sine_wave(200.0, duration_s=2.0)
        stats = analyze_pitch(audio, TARGET_SR)
        assert len(stats["f0_times"]) == len(stats["f0_values"])

    def test_classification_is_valid(self):
        audio = _sine_wave(300.0)  # Should be "high"
        stats = analyze_pitch(audio, TARGET_SR)
        assert stats["classification"] in {"high", "mid", "low", "unknown"}


class TestPlotPitchContour:
    def test_creates_png(self):
        audio = _sine_wave(250.0, duration_s=2.0)
        stats = analyze_pitch(audio, TARGET_SR)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        try:
            plot_pitch_contour(stats, tmp.name, title="Test Plot")
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
        finally:
            os.unlink(tmp.name)

    def test_creates_png_for_silent_audio(self):
        audio = np.zeros(TARGET_SR, dtype=np.float32)
        stats = analyze_pitch(audio, TARGET_SR)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        try:
            plot_pitch_contour(stats, tmp.name)
            assert os.path.exists(tmp.name)
        finally:
            os.unlink(tmp.name)
