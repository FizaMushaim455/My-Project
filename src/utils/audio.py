"""
audio.py — Text-to-speech audio synthesis wrapper.

Wraps gTTS (Google Text-to-Speech) and playsound so the rest of the codebase
can call a simple ``play_speech()`` function without worrying about file
management or fallback handling.
"""

import logging
import os
from pathlib import Path

from src.config.settings import AUDIO_OUTPUT_FILE, TTS_LANGUAGE, TTS_SLOW

logger = logging.getLogger(__name__)


def play_speech(
    text: str,
    lang: str = TTS_LANGUAGE,
    slow: bool = TTS_SLOW,
    output_file: str = AUDIO_OUTPUT_FILE,
) -> None:
    """Convert text to speech and play it through the system audio output.

    Uses gTTS to synthesise the audio and ``playsound`` to play it.
    Falls back to ``os.startfile`` (Windows) / ``os.system`` if playsound
    raises an exception.

    Args:
        text: The text to synthesise.
        lang: BCP-47 language code (default ``'ur'`` = Urdu).
        slow: Whether to use gTTS slow mode (default ``False``).
        output_file: Path where the temporary MP3 file is written.

    Raises:
        ValueError: If ``text`` is empty or whitespace-only.
    """
    from gtts import gTTS  # imported here so the module loads without gTTS installed
    from playsound import playsound  # same

    if not text or not text.strip():
        raise ValueError("play_speech: text must be a non-empty string.")

    logger.info("Synthesising speech for text: '%s' (lang=%s)", text, lang)

    # Remove stale file to avoid permission errors on Windows
    output_path = Path(output_file)
    if output_path.exists():
        try:
            output_path.unlink()
        except OSError as exc:
            logger.warning("Could not remove previous audio file: %s", exc)

    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(output_file)
    logger.info("Audio saved to '%s'; playing now.", output_file)

    try:
        playsound(output_file)
    except Exception as exc:  # noqa: BLE001
        logger.warning("playsound failed (%s); attempting OS fallback.", exc)
        os.startfile(output_file)  # Windows fallback
