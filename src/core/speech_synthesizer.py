"""
speech_synthesizer.py — Orchestrates translation + text-to-speech output.

Provides a simple high-level class that takes a recognised English string,
translates it to Urdu, and plays audio — hiding all gTTS / playsound details.
"""

import logging

from src.config.settings import AUDIO_OUTPUT_FILE, TTS_LANGUAGE, TTS_SLOW
from src.utils.audio import play_speech
from src.utils.translation import translate_to_urdu

logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    """Translates recognised ASL text to Urdu and synthesises speech.

    Example usage::

        synth = SpeechSynthesizer()
        synth.speak("HELLO HOW ARE YOU")

    Args:
        lang: BCP-47 language code for TTS output (default ``'ur'`` = Urdu).
        slow: Whether to use gTTS slow mode.
        output_file: Path for the temporary audio MP3 file.
    """

    def __init__(
        self,
        lang: str = TTS_LANGUAGE,
        slow: bool = TTS_SLOW,
        output_file: str = AUDIO_OUTPUT_FILE,
    ) -> None:
        self.lang = lang
        self.slow = slow
        self.output_file = output_file

    def speak(self, english_text: str) -> str:
        """Translate English text to Urdu and play it as speech.

        Args:
            english_text: Recognised English word(s) from ASL detection.

        Returns:
            The translated Urdu text that was spoken.

        Raises:
            ValueError: If ``english_text`` is empty after stripping.
        """
        english_text = english_text.strip()
        if not english_text:
            raise ValueError("speak(): received empty text — nothing to synthesise.")

        urdu_text = translate_to_urdu(english_text)
        logger.info(
            "English: '%s'  →  Urdu: '%s'", english_text, urdu_text
        )

        if urdu_text.strip():
            play_speech(
                urdu_text,
                lang=self.lang,
                slow=self.slow,
                output_file=self.output_file,
            )
        else:
            logger.warning("Translation produced an empty string; skipping TTS.")

        return urdu_text
