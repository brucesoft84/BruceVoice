"""Speaker diarization using pyannote.audio pipeline."""

import logging
import os
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_pipeline_cache: dict = {}
_plda_patched = False


def _patch_get_plda() -> None:
    """Monkey-patch pyannote's get_plda to return None on download failures.

    pyannote.audio 4.x unconditionally tries to fetch PLDA weights from the
    restricted pyannote/speaker-diarization-community-1 model in __init__,
    even when AgglomerativeClustering is used (which ignores PLDA entirely).
    This patch makes the failure silent so that speaker-diarization-3.1 loads
    with AgglomerativeClustering without needing community-1 access.
    """
    global _plda_patched
    if _plda_patched:
        return
    try:
        from pyannote.audio.pipelines.utils import getter as _getter
        _orig = _getter.get_plda

        def _safe_get_plda(plda, token=None, cache_dir=None):
            try:
                return _orig(plda, token=token, cache_dir=cache_dir)
            except Exception as exc:
                logger.warning("PLDA download skipped (%s). AgglomerativeClustering will be used.", exc)
                return None

        _getter.get_plda = _safe_get_plda
        # Also patch the reference inside the speaker_diarization module
        from pyannote.audio.pipelines import speaker_diarization as _sd_mod
        _sd_mod.get_plda = _safe_get_plda
        _plda_patched = True
        logger.debug("Patched get_plda to suppress community-1 download errors")
    except Exception as exc:
        logger.warning("Could not patch get_plda: %s", exc)


def _get_hf_token() -> Optional[str]:
    """Return Hugging Face access token from environment or None."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def load_diarization_pipeline(device: str = "auto"):
    """Load (or return cached) pyannote speaker-diarization pipeline.

    Requires a Hugging Face token with access to pyannote/speaker-diarization-3.1.
    Set HF_TOKEN or HUGGINGFACE_TOKEN environment variable.

    Args:
        device: 'auto', 'cpu', or 'cuda'.

    Returns:
        pyannote Pipeline instance.
    """
    resolved_device = _resolve_device(device)
    cache_key = ("diarization", resolved_device)

    if cache_key in _pipeline_cache:
        logger.info("Using cached diarization pipeline")
        return _pipeline_cache[cache_key]

    token = _get_hf_token()
    if token is None:
        logger.warning(
            "HF_TOKEN not set. If model access is restricted, set the environment variable."
        )

    # Explicitly authenticate the HF hub session so gated models work
    if token:
        import huggingface_hub
        huggingface_hub.login(token=token, add_to_git_credential=False)

    # torchaudio 2.4+ removed list_audio_backends(); speechbrain (used by pyannote 4.x)
    # calls it at import time. Restore it as a no-op BEFORE importing pyannote.
    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["sox", "soundfile"]

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise ImportError(
            "pyannote.audio is required for diarization. "
            "Install it with: pip install pyannote.audio"
        ) from exc

    # pyannote 4.x unconditionally calls get_plda() in __init__, which tries to download
    # pyannote/speaker-diarization-community-1 (a restricted model).
    # AgglomerativeClustering (used by speaker-diarization-3.1) ignores PLDA entirely,
    # so we safely patch get_plda to return None when the download fails.
    _patch_get_plda()

    logger.info("Loading pyannote/speaker-diarization-3.1 on %s", resolved_device)
    diar_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )
    diar_pipeline.to(torch.device(resolved_device))

    _pipeline_cache[cache_key] = diar_pipeline
    return diar_pipeline


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def diarize_speakers(
    audio: np.ndarray,
    sr: int,
    device: str = "auto",
    num_speakers: Optional[int] = None,
) -> list:
    """Run speaker diarization on the audio and return speaker segments.

    Args:
        audio: Audio samples as float32 at 16kHz.
        sr: Sample rate (should be 16000).
        device: Compute device.
        num_speakers: Optional expected speaker count hint for the model.

    Returns:
        List of segment dicts:
        [{"speaker": "SPEAKER_00", "start": float, "end": float}, ...]
        Sorted by start time.
    """
    import io

    import soundfile as sf

    diar_pipeline = load_diarization_pipeline(device)

    # pyannote expects a dict with waveform tensor + sample_rate, or a file path.
    # We pass a tensor directly to avoid writing a temp file.
    logger.info("Running speaker diarization...")

    waveform = torch.from_numpy(audio).unsqueeze(0)  # shape: (1, samples)
    input_dict = {"waveform": waveform, "sample_rate": sr}

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    with torch.no_grad():
        result = diar_pipeline(input_dict, **kwargs)

    # pyannote 4.x returns a DiarizeOutput dataclass; 3.x returns an Annotation directly.
    if hasattr(result, "speaker_diarization"):
        diarization = result.speaker_diarization
    else:
        diarization = result

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "speaker": speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            }
        )

    segments.sort(key=lambda x: x["start"])

    speaker_ids = sorted({s["speaker"] for s in segments})
    logger.info(
        "Diarization complete: %d speakers, %d segments",
        len(speaker_ids),
        len(segments),
    )
    return segments


def align_transcript_to_speakers(
    segments: list,
    timed_chunks: list,
) -> dict:
    """Align timestamped transcript chunks to diarization speaker segments.

    Works best with word-level timestamps (the default from
    transcribe_with_timestamps). Each chunk is assigned to the speaker whose
    diarization segment covers the chunk midpoint; consecutive chunks from the
    same speaker are merged into utterances.

    Falls back to max-overlap assignment for longer chunks or gaps between
    diarization segments.

    Args:
        segments: Diarization output from diarize_speakers().
        timed_chunks: List of {"text": str, "timestamp": (start, end)}.
                      Can be word-level or segment-level.

    Returns:
        Dict mapping speaker_id -> list of
        {"text": str, "start": float, "end": float}.
    """
    from collections import defaultdict

    if not timed_chunks or not segments:
        return {}

    # ── Step 1: tag each chunk with a speaker ──────────────────────────────
    tagged: list = []  # [(speaker, text, start, end), ...]

    for chunk in timed_chunks:
        text = chunk.get("text", "").strip()
        ts = chunk.get("timestamp", (None, None))
        if not text or ts[0] is None:
            continue

        start = ts[0]
        end = ts[1] if ts[1] is not None else ts[0] + 0.3
        mid = (start + end) / 2.0

        # Midpoint lookup: precise for word-level (~0.2 s) tokens
        speaker = _find_speaker_at_time(segments, mid)
        # Fallback: max-overlap (handles gaps or long segment-level chunks)
        if speaker is None:
            speaker = _find_speaker(segments, start, end)

        if speaker:
            tagged.append((speaker, text, start, end))

    if not tagged:
        return {}

    # ── Step 2: group consecutive same-speaker chunks into utterances ───────
    speaker_transcripts: dict = defaultdict(list)

    i = 0
    while i < len(tagged):
        spk, text, utt_start, utt_end = tagged[i]
        words = [text]

        j = i + 1
        while j < len(tagged) and tagged[j][0] == spk:
            words.append(tagged[j][1])
            utt_end = tagged[j][3]
            j += 1

        speaker_transcripts[spk].append(
            {"text": " ".join(words), "start": utt_start, "end": utt_end}
        )
        i = j

    return dict(speaker_transcripts)


# ── Role identification keywords (Vietnamese call-center / MEDLATEC context) ───
_AGENT_KEYWORDS = [
    # Generic call-center agent phrases
    "xin nghe", "tên là", "bên em", "chúng tôi", "dịch vụ",
    "cảm ơn anh", "cảm ơn chị", "chào anh", "chào chị",
    "dạ vâng", "vâng ạ", "xin chào", "để xin nghe",
    "triển khai", "khu vực", "hỗ trợ",
    # MEDLATEC brand variants (phonetically transcribed by PhoWhisper)
    "medlatec", "mê la tét", "may la tét", "me la tec", "mê la tec",
    # MEDLATEC healthcare agent phrases
    "kết quả xét nghiệm", "kết quả của anh", "kết quả của chị",
    "đặt lịch lấy mẫu", "lấy mẫu tại nhà", "cử nhân viên",
    "nhân viên sẽ đến", "xét nghiệm tại nhà", "gói xét nghiệm",
    "anh chị cần xét nghiệm", "chị cần xét nghiệm", "anh cần xét nghiệm",
    "đã ghi nhận", "đã tiếp nhận", "sẽ gọi lại", "bên medlatec",
    "hệ thống medlatec", "cơ sở medlatec",
]
_CUSTOMER_KEYWORDS = [
    # Generic customer phrases
    "cho tôi hỏi", "cho anh hỏi", "cho chị hỏi",
    "tôi muốn", "tôi cần", "có không", "được không",
    "làm được không", "như thế nào", "em ơi", "chị ơi",
    # Healthcare-specific customer phrases
    "kết quả của tôi", "kết quả của anh", "kết quả ra chưa",
    "lấy kết quả", "có kết quả chưa", "bao giờ có kết quả",
    "đặt lịch khám", "đặt lịch xét nghiệm", "lấy mẫu tại nhà",
    "xét nghiệm gì", "chi phí bao nhiêu", "giá bao nhiêu",
    "địa chỉ ở đâu", "cơ sở nào gần nhất",
]


def identify_speaker_roles(
    speaker_segments: list,
    speaker_transcripts: dict,
) -> dict:
    """Identify which speaker is the call-center agent and which is the customer.

    Uses two complementary heuristics:
    1. **First speaker = agent** — call-center agents answer the phone, so
       the speaker whose first segment starts earliest is biased toward agent.
    2. **Keyword scoring** — agent phrases (greetings, closings, service
       language) and customer phrases (questions, requests) shift the score.

    Args:
        speaker_segments: Output of diarize_speakers().
        speaker_transcripts: Dict mapping speaker_id -> list of chunk dicts
            (from align_transcript_to_speakers) or plain strings.

    Returns:
        Dict mapping speaker_id -> "agent" | "customer".
    """
    speaker_ids = sorted({s["speaker"] for s in speaker_segments})
    if not speaker_ids:
        return {}

    # Heuristic 1: earliest first-segment → likely agent (answered the call)
    first_start: dict = {}
    for seg in sorted(speaker_segments, key=lambda x: x["start"]):
        if seg["speaker"] not in first_start:
            first_start[seg["speaker"]] = seg["start"]
    earliest = min(first_start, key=first_start.get)

    # Heuristic 2: keyword score (positive = agent, negative = customer)
    scores = {spk: 0 for spk in speaker_ids}
    scores[earliest] += 2  # prior toward earliest = agent

    for spk_id in speaker_ids:
        chunks = speaker_transcripts.get(spk_id, [])
        if isinstance(chunks, list):
            text = " ".join(
                c["text"] if isinstance(c, dict) else str(c) for c in chunks
            ).lower()
        else:
            text = str(chunks).lower()

        for kw in _AGENT_KEYWORDS:
            if kw in text:
                scores[spk_id] += 1
        for kw in _CUSTOMER_KEYWORDS:
            if kw in text:
                scores[spk_id] -= 1

    sorted_spk = sorted(speaker_ids, key=lambda s: scores[s], reverse=True)
    roles = {spk: ("agent" if i == 0 else "customer") for i, spk in enumerate(sorted_spk)}
    logger.info("Role identification: %s", roles)
    return roles


def verify_speaker_transcripts(
    audio: np.ndarray,
    sr: int,
    speaker_segments: list,
    model_size: str = "large",
    device: str = "auto",
) -> dict:
    """Re-transcribe each speaker's audio for cross-verification.

    Extracts and concatenates the audio portions belonging to each diarized
    speaker, then transcribes that audio independently.  Single-speaker audio
    is significantly easier for PhoWhisper to transcribe accurately, so this
    step both improves per-speaker transcript quality and acts as a sanity
    check on the diarization alignment.

    Args:
        audio: Full recording as float32 at 16 kHz.
        sr: Sample rate (should be 16000).
        speaker_segments: Output of diarize_speakers().
        model_size: PhoWhisper model size.
        device: Compute device.

    Returns:
        Dict mapping speaker_id -> verified transcript string.
    """
    from .transcribe import transcribe

    speaker_ids = sorted({s["speaker"] for s in speaker_segments})
    verified: dict = {}
    silence = np.zeros(int(0.25 * sr), dtype=np.float32)  # 250 ms gap

    for spk_id in speaker_ids:
        segs = [s for s in speaker_segments if s["speaker"] == spk_id]

        parts = []
        for seg in segs:
            # Add a small buffer so word boundaries aren't clipped
            start = max(0, int((seg["start"] - 0.05) * sr))
            end = min(len(audio), int((seg["end"] + 0.05) * sr))
            seg_audio = audio[start:end]

            # Skip transition artefacts shorter than 0.3 s
            if len(seg_audio) >= int(0.3 * sr):
                parts.append(seg_audio)

        if not parts:
            verified[spk_id] = ""
            continue

        # Interleave parts with short silence so the model doesn't merge turns
        chunks: list = []
        for i, part in enumerate(parts):
            chunks.append(part)
            if i < len(parts) - 1:
                chunks.append(silence)
        combined = np.concatenate(chunks)

        logger.info(
            "Verifying transcript for %s: %.1f s of audio across %d segment(s)",
            spk_id,
            len(combined) / sr,
            len(parts),
        )
        text = transcribe(combined, sr, model_size=model_size, device=device)
        verified[spk_id] = text.strip()

    return verified


def _find_speaker_at_time(segments: list, t: float) -> Optional[str]:
    """Return the speaker whose segment covers time t, or None."""
    for seg in segments:
        if seg["start"] <= t <= seg["end"]:
            return seg["speaker"]
    return None


def _find_speaker(segments: list, start: float, end: float) -> Optional[str]:
    """Find the speaker with the maximum overlap with a given time range."""
    if end is None:
        end = start + 0.5

    best_speaker = None
    best_overlap = 0.0

    for seg in segments:
        overlap_start = max(seg["start"], start)
        overlap_end = min(seg["end"], end)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]

    return best_speaker
