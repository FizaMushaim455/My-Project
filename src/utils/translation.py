"""
translation.py — English-to-Urdu word translation helper.

Provides a simple dictionary-based mapping extracted from the notebook.
Words not found in the dictionary are passed through unchanged.
"""

import logging
from typing import Dict

from src.config.settings import ENGLISH_TO_URDU_DICT

logger = logging.getLogger(__name__)


def translate_to_urdu(
    english_text: str,
    dictionary: Dict[str, str] = ENGLISH_TO_URDU_DICT,
) -> str:
    """Translate an English word/sentence to Urdu using a lookup dictionary.

    Each word in ``english_text`` is looked up in ``dictionary`` (case-
    insensitive).  Words not found are left in their original English form so
    no information is lost.

    Args:
        english_text: The recognised English text (e.g. ``"HELLO HOW ARE YOU"``).
        dictionary: Word-level English→Urdu mapping.  Defaults to the project
            dictionary from ``settings.py``.

    Returns:
        A space-joined string of translated (or unchanged) words.

    Examples:
        >>> translate_to_urdu("HELLO HOW ARE YOU")
        'ہیلو کیسے ہیں آپ'
        >>> translate_to_urdu("UNKNOWN WORD")
        'unknown word'
    """
    cleaned = english_text.strip().lower()
    if not cleaned:
        logger.debug("translate_to_urdu received empty string; returning empty.")
        return ""

    urdu_words = []
    for word in cleaned.split():
        translated = dictionary.get(word, word)
        urdu_words.append(translated)

    result = " ".join(urdu_words)
    logger.debug("Translated '%s' → '%s'", english_text, result)
    return result
