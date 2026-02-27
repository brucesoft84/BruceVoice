"""Unit tests for audio loading utilities."""

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from voice_analyzer.audio import TARGET_SR, chunk_audio, load_audio, save_wav


def _make_wav(duration_s: float = 1.0, sr: int = TARGET_SR) -> str:
    """Create a temporary WAV file with a sine wave and return its path."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    tmp.close()
    return tmp.name


class TestLoadAudio:
    def test_loads_wav(self):
        path = _make_wav()
        try:
            audio, sr = load_audio(path)
            assert sr == TARGET_SR
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            assert len(audio) > 0
        finally:
            os.unlink(path)

    def test_resamples_to_16k(self):
        # Create a 22050 Hz WAV
        sr_orig = 22050
        t = np.linspace(0, 1.0, sr_orig, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, sr_orig)
        tmp.close()
        try:
            out_audio, out_sr = load_audio(tmp.name)
            assert out_sr == TARGET_SR
        finally:
            os.unlink(tmp.name)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/path/audio.wav")

    def test_unsupported_format(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False)
        tmp.close()
        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                load_audio(tmp.name)
        finally:
            os.unlink(tmp.name)


class TestChunkAudio:
    def test_single_chunk_for_short_audio(self):
        audio = np.zeros(TARGET_SR * 10, dtype=np.float32)  # 10s
        chunks = chunk_audio(audio, TARGET_SR, chunk_sec=30)
        assert len(chunks) == 1
        assert chunks[0][0] == pytest.approx(0.0)

    def test_multiple_chunks_for_long_audio(self):
        audio = np.zeros(TARGET_SR * 75, dtype=np.float32)  # 75s
        chunks = chunk_audio(audio, TARGET_SR, chunk_sec=30)
        assert len(chunks) == 3  # 30s, 30s, 15s

    def test_chunk_start_times(self):
        audio = np.zeros(TARGET_SR * 60, dtype=np.float32)  # 60s
        chunks = chunk_audio(audio, TARGET_SR, chunk_sec=30)
        assert chunks[0][0] == pytest.approx(0.0)
        assert chunks[1][0] == pytest.approx(30.0)


class TestSaveWav:
    def test_saves_and_reloads(self):
        audio = np.random.uniform(-0.5, 0.5, TARGET_SR).astype(np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            save_wav(audio, TARGET_SR, tmp.name)
            loaded, sr = sf.read(tmp.name)
            assert sr == TARGET_SR
            np.testing.assert_allclose(loaded, audio, atol=1e-5)
        finally:
            os.unlink(tmp.name)
