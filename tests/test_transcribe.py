"""Unit tests for transcription utilities (model loading logic, not inference)."""

import pytest

from voice_analyzer.transcribe import MODEL_MAP, _get_device


class TestGetDevice:
    def test_cpu_explicit(self):
        assert _get_device("cpu") == "cpu"

    def test_cuda_explicit(self):
        assert _get_device("cuda") == "cuda"

    def test_auto_returns_string(self):
        # auto should return either 'cpu' or 'cuda' depending on environment
        result = _get_device("auto")
        assert result in {"cpu", "cuda"}


class TestModelMap:
    def test_all_sizes_present(self):
        for size in ("tiny", "base", "small", "medium", "large"):
            assert size in MODEL_MAP

    def test_model_ids_are_vinai(self):
        for model_id in MODEL_MAP.values():
            assert model_id.startswith("vinai/PhoWhisper")

    def test_load_model_raises_on_unknown_size(self):
        from voice_analyzer.transcribe import load_model

        with pytest.raises(ValueError, match="Unknown model size"):
            load_model("xlarge")
