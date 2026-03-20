"""
asl_recognizer.py — High-level ASL sign recogniser class.

Wraps model loading and per-frame prediction into a reusable object
so both the real-time pipeline and any future interface can share it.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.config.settings import IMG_SIZE, LABELS_DICT, MIN_VALUE, MODEL_PATH, ROI
from src.models.model_utils import load_model
from src.utils.image_processing import preprocess_frame

logger = logging.getLogger(__name__)


class ASLRecognizer:
    """Loads the pre-trained ASL CNN and predicts hand gesture labels.

    Example usage::

        recognizer = ASLRecognizer()
        recognizer.load()
        label, confidence = recognizer.predict(frame)

    Args:
        model_path: Path to the ``.h5`` model file.
        labels_dict: Mapping from class index → ASL character.
        img_size: Side length of the square input image.
        min_value: Otsu threshold minimum.
        roi: ``(x1, y1, x2, y2)`` region of interest in camera frames.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        labels_dict: dict = LABELS_DICT,
        img_size: int = IMG_SIZE,
        min_value: int = MIN_VALUE,
        roi: Tuple[int, int, int, int] = ROI,
    ) -> None:
        self.model_path = model_path
        self.labels_dict = labels_dict
        self.img_size = img_size
        self.min_value = min_value
        self.roi = roi
        self._model: Optional[object] = None

    def load(self) -> None:
        """Load the Keras model from disk.

        Must be called before :meth:`predict`.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        self._model = load_model(self.model_path)
        logger.info("ASLRecognizer: model loaded from '%s'.", self.model_path)

    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        """Predict the ASL gesture label for a single camera frame.

        Args:
            frame: Raw BGR frame from ``cv2.VideoCapture``.

        Returns:
            Tuple of ``(label, confidence)`` where ``label`` is the predicted
            ASL character (e.g. ``'A'``) and ``confidence`` is in ``[0, 1]``.

        Raises:
            RuntimeError: If :meth:`load` has not been called first.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call ASLRecognizer.load() first.")

        _, model_input = preprocess_frame(
            frame, self.roi, self.img_size, self.min_value
        )
        result = self._model.predict(model_input, verbose=0)
        idx = int(np.argmax(result, axis=1)[0])
        confidence = float(result[0][idx])
        label = self.labels_dict[idx]

        logger.debug("Predicted: '%s' (confidence=%.2f)", label, confidence)
        return label, confidence
