"""Output formatting helpers: console (colored), JSON, text file."""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"
_DIM = "\033[2m"


def _c(text: str, color: str) -> str:
    """Wrap text in ANSI color code."""
    return f"{color}{text}{_RESET}"


def print_results(
    transcript: str,
    pitch_stats: dict,
    speaker_segments: list,
    speaker_transcripts: dict,
) -> None:
    """Print a structured, colored analysis report to stdout.

    Args:
        transcript: Full Vietnamese transcript string.
        pitch_stats: Output of analyze_pitch().
        speaker_segments: Diarization segments list.
        speaker_transcripts: Speaker -> transcript chunks mapping.
    """
    sep = _c("─" * 60, _DIM)

    # ── Transcript ──────────────────────────────────────────────
    print(f"\n{_c('TRANSCRIPT', _BOLD + _CYAN)}")
    print(sep)
    print(transcript or _c("(no transcript)", _DIM))

    # ── Pitch ────────────────────────────────────────────────────
    print(f"\n{_c('PITCH ANALYSIS', _BOLD + _GREEN)}")
    print(sep)
    p = pitch_stats
    mean_str = f"{p['mean_hz']:.1f} Hz"
    print(f"  Mean:           {_c(mean_str, _GREEN)}")
    print(f"  Min:            {p['min_hz']:.1f} Hz")
    print(f"  Max:            {p['max_hz']:.1f} Hz")
    print(f"  Median:         {p['median_hz']:.1f} Hz")
    print(f"  Std Dev:        {p['std_hz']:.1f} Hz")
    cls = p["classification"].upper()
    cls_color = _YELLOW if cls == "HIGH" else (_GREEN if cls == "MID" else _CYAN)
    print(f"  Classification: {_c(cls, cls_color)}")

    # ── Speaker Diarization ──────────────────────────────────────
    print(f"\n{_c('SPEAKER DIARIZATION', _BOLD + _MAGENTA)}")
    print(sep)

    if not speaker_segments:
        print(_c("  (diarization not run)", _DIM))
    else:
        speaker_ids = sorted({s["speaker"] for s in speaker_segments})
        print(f"  Detected speakers: {_c(str(len(speaker_ids)), _BOLD)}")
        print()

        # Timeline
        print(f"  {_c('Timeline:', _BOLD)}")
        for seg in speaker_segments:
            bar_len = max(1, int((seg["end"] - seg["start"]) * 4))
            bar = "█" * bar_len
            print(
                f"    [{seg['start']:6.1f}s – {seg['end']:6.1f}s]  "
                f"{_c(seg['speaker'], _MAGENTA)}  {_c(bar, _DIM)}"
            )

        # Per-speaker transcripts
        if speaker_transcripts:
            print(f"\n  {_c('Speaker Transcripts:', _BOLD)}")
            for spk_id in sorted(speaker_transcripts):
                chunks = speaker_transcripts[spk_id]
                combined = " ".join(c["text"] for c in chunks)
                print(f"\n  {_c(spk_id, _MAGENTA + _BOLD)}")
                print(f"    {combined}")

    print()


def build_conversation_timeline(
    aligned_transcripts: dict,
    pitch_stats: dict,
    roles: dict,
) -> list:
    """Build a chronological conversation timeline with pitch annotation.

    Each entry is one contiguous speaker turn (utterance) with its start/end
    time, role label, average F0, and a high-tone flag.

    An utterance is flagged **high_tone** when its average pitch exceeds
    115 % of the recording's overall mean F0 — a speaker-relative threshold
    that works equally well for male and female voices.

    Args:
        aligned_transcripts: Output of align_transcript_to_speakers() —
            {speaker_id: [{text, start, end}, ...]} with precise timestamps.
        pitch_stats: Output of analyze_pitch() (must still contain f0_times /
            f0_values before they are stripped).
        roles: Output of identify_speaker_roles() — {speaker_id: role_str}.

    Returns:
        List of utterance dicts sorted by start time:
        [{speaker, role, start, end, text, avg_pitch_hz, high_tone}, ...]
    """
    from .pitch import avg_pitch_for_window

    mean_hz = pitch_stats.get("mean_hz", 200.0)
    high_tone_threshold = mean_hz * 1.15  # 15 % above recording mean

    utterances = []
    for spk_id, chunks in aligned_transcripts.items():
        role = (roles or {}).get(spk_id, "unknown")
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            start = chunk.get("start", 0.0)
            end = chunk.get("end", 0.0)
            avg_hz = avg_pitch_for_window(pitch_stats, start, end)
            utterances.append(
                {
                    "speaker": spk_id,
                    "role": role,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": text,
                    "avg_pitch_hz": avg_hz,
                    "high_tone": avg_hz is not None and avg_hz > high_tone_threshold,
                }
            )

    utterances.sort(key=lambda x: x["start"])
    return utterances


def build_json_output(
    transcript: str,
    pitch_stats: dict,
    speaker_segments: list,
    speaker_transcripts: dict,
    roles: dict = None,
    conversation: list = None,
) -> dict:
    """Build the full analysis result as a serialisable dict.

    Args:
        transcript: Full transcript string.
        pitch_stats: Output of analyze_pitch() (f0_times/values stripped for brevity).
        speaker_segments: Diarization segments.
        speaker_transcripts: Speaker -> transcript chunks.

    Returns:
        Dict matching the required JSON schema.
    """
    pitch_summary = {
        "mean_hz": pitch_stats.get("mean_hz", 0.0),
        "min_hz": pitch_stats.get("min_hz", 0.0),
        "max_hz": pitch_stats.get("max_hz", 0.0),
        "median_hz": pitch_stats.get("median_hz", 0.0),
        "std_hz": pitch_stats.get("std_hz", 0.0),
        "classification": pitch_stats.get("classification", "unknown"),
    }

    speakers_out = []
    speaker_ids = sorted({s["speaker"] for s in speaker_segments}) if speaker_segments else []
    for spk_id in speaker_ids:
        segs = [s for s in speaker_segments if s["speaker"] == spk_id]
        chunks = speaker_transcripts.get(spk_id, [])
        spk_transcript = " ".join(c["text"] for c in chunks)
        role = (roles or {}).get(spk_id, "unknown")
        speakers_out.append(
            {
                "id": spk_id,
                "role": role,
                "segments": [{"start": s["start"], "end": s["end"]} for s in segs],
                "transcript": spk_transcript,
            }
        )

    result = {
        "transcript": transcript,
        "pitch": pitch_summary,
        "speakers": speakers_out,
    }
    if conversation is not None:
        result["conversation"] = conversation
    return result


def save_text(text: str, path: str) -> None:
    """Write transcript to a plain-text file."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    logger.info("Transcript saved: %s", path)


def save_json(data: dict, path: str) -> None:
    """Write analysis dict to a JSON file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    logger.info("JSON report saved: %s", path)
