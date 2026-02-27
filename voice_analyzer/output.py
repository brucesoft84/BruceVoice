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


def build_json_output(
    transcript: str,
    pitch_stats: dict,
    speaker_segments: list,
    speaker_transcripts: dict,
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
        speakers_out.append(
            {
                "id": spk_id,
                "segments": [{"start": s["start"], "end": s["end"]} for s in segs],
                "transcript": spk_transcript,
            }
        )

    return {
        "transcript": transcript,
        "pitch": pitch_summary,
        "speakers": speakers_out,
    }


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
