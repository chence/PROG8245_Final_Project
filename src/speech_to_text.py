from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from src.config import get_config


@dataclass
class TranscriptionResult:
    text: str
    success: bool
    error: str | None = None


def transcribe_audio(audio_path: str | Path) -> TranscriptionResult:
    config = get_config()
    if not config.openai_api_key:
        return TranscriptionResult(
            text="",
            success=False,
            error="OPENAI_API_KEY is not configured, so audio transcription is unavailable.",
        )

    client = OpenAI(api_key=config.openai_api_key)
    path = Path(audio_path)
    if not path.exists():
        return TranscriptionResult(text="", success=False, error="Audio file not found.")

    try:
        with open(path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=config.openai_transcription_model,
                file=audio_file,
                response_format="text",
                prompt=(
                    "The transcript is for a multilingual medical information chatbot. "
                    "Prefer accurate medication names, symptom names, and short phrases."
                ),
            )
        text = transcript if isinstance(transcript, str) else getattr(transcript, "text", "").strip()
        if not text:
            return TranscriptionResult(text="", success=False, error="No speech was detected.")
        return TranscriptionResult(text=text, success=True)
    except Exception as exc:
        return TranscriptionResult(text="", success=False, error=f"Transcription failed: {exc}")
