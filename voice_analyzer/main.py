"""CLI entry point for voice-analyzer."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .audio import load_audio
from .diarize import align_transcript_to_speakers, diarize_speakers
from .output import build_json_output, print_results, save_json, save_text
from .pitch import analyze_pitch, plot_pitch_contour
from .transcribe import transcribe, transcribe_with_timestamps

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("voice_analyzer")


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command(name="voice-analyzer")
@click.version_option(version=__version__, prog_name="voice-analyzer")
@click.argument("audio_file", metavar="AUDIO_FILE")
@click.option(
    "--output-text",
    "output_text",
    default=None,
    metavar="TEXT_FILE",
    help="Save transcript to a text file (default: stdout).",
)
@click.option(
    "--output-json",
    "output_json",
    default=None,
    metavar="JSON_FILE",
    help="Save full analysis (transcript + pitch + diarization) as JSON.",
)
@click.option(
    "--output-pitch",
    "output_pitch",
    default=None,
    metavar="PNG_FILE",
    help="Save pitch contour plot as PNG.",
)
@click.option(
    "--model-size",
    "model_size",
    type=click.Choice(["tiny", "base", "small", "medium", "large"], case_sensitive=False),
    default="base",
    show_default=True,
    help="PhoWhisper model size.",
)
@click.option(
    "--num-speakers",
    "num_speakers",
    default=None,
    type=int,
    metavar="INT",
    help="Expected number of speakers (optional hint for diarization).",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Compute device for models.",
)
@click.option(
    "--skip-diarization",
    "skip_diarization",
    is_flag=True,
    default=False,
    help="Skip speaker diarization (faster, no HF token required).",
)
def cli(
    audio_file: str,
    output_text: Optional[str],
    output_json: Optional[str],
    output_pitch: Optional[str],
    model_size: str,
    num_speakers: Optional[int],
    device: str,
    skip_diarization: bool,
) -> None:
    """Analyse a Vietnamese audio file: transcription, pitch, and speaker diarization.

    AUDIO_FILE  Path to a WAV, MP3, or M4A audio file.

    \b
    Examples:
      voice-analyzer recording.wav
      voice-analyzer interview.mp3 --output-json report.json --output-pitch pitch.png
      voice-analyzer call.m4a --model-size large --num-speakers 2
    """
    # ── 1. Load audio ──────────────────────────────────────────────────────────
    try:
        logger.info("Step 1/4: Loading audio")
        audio, sr = load_audio(audio_file)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # ── 2. Transcribe ──────────────────────────────────────────────────────────
    logger.info("Step 2/4: Transcribing speech")
    if skip_diarization and output_json is None:
        # Fast path: plain transcript only
        transcript = transcribe(audio, sr, model_size=model_size, device=device)
        timed_chunks = []
    else:
        # We need timestamps to align speaker segments
        timed_chunks = transcribe_with_timestamps(audio, sr, model_size=model_size, device=device)
        transcript = " ".join(c["text"] for c in timed_chunks)

    # ── 3. Pitch analysis ──────────────────────────────────────────────────────
    logger.info("Step 3/4: Analysing pitch")
    pitch_stats = analyze_pitch(audio, sr)

    if output_pitch:
        plot_pitch_contour(
            pitch_stats,
            output_pitch,
            title=f"Pitch Contour – {Path(audio_file).name}",
        )

    # ── 4. Diarization ─────────────────────────────────────────────────────────
    speaker_segments: list = []
    speaker_transcripts: dict = {}

    if not skip_diarization:
        logger.info("Step 4/4: Diarizing speakers")
        try:
            speaker_segments = diarize_speakers(
                audio, sr, device=device, num_speakers=num_speakers
            )
            speaker_transcripts = align_transcript_to_speakers(speaker_segments, timed_chunks)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Diarization failed: %s", exc)
            click.echo(
                f"Warning: diarization skipped ({exc}). "
                "Set HF_TOKEN env var and accept pyannote model terms on Hugging Face.",
                err=True,
            )
    else:
        logger.info("Step 4/4: Diarization skipped (--skip-diarization)")

    # ── Outputs ────────────────────────────────────────────────────────────────
    if output_text:
        save_text(transcript, output_text)
    else:
        # Print transcript to stdout if no file requested
        click.echo(transcript)

    if output_json:
        data = build_json_output(transcript, pitch_stats, speaker_segments, speaker_transcripts)
        save_json(data, output_json)

    # Always print the structured console report
    print_results(transcript, pitch_stats, speaker_segments, speaker_transcripts)

    click.echo(f"\nDone. Analysed: {audio_file}", err=True)


if __name__ == "__main__":
    cli()
