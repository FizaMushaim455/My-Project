"""
model_utils.py — Helpers for saving and loading Keras models.

Centralises model persistence logic so training and inference code
don't need to know file formats or paths.
"""

import logging
from pathlib import Path

from tensorflow.keras.models import Sequential, load_model as _keras_load  # type: ignore

from src.config.settings import MODEL_PATH

logger = logging.getLogger(__name__)


def save_model(model: Sequential, path: Path = MODEL_PATH) -> None:
    """Persist a Keras model to disk in HDF5 format.

    Args:
        model: Compiled Keras model to save.
        path: Destination file path (should end with ``.h5``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    logger.info("Model saved to '%s'.", path)


def load_model(path: Path = MODEL_PATH) -> Sequential:
    """Load a Keras model from an HDF5 file.

    Args:
        path: Path to the ``.h5`` model file.

    Returns:
        Loaded Keras model, ready for inference.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    from src.models.asl_classifier import build_model

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Train the model first with: python main.py train"
        )

    try:
        # Standard load (works for models saved in the same Keras version)
        model = _keras_load(str(path))
        logger.info("Model loaded from '%s' using standard loader.", path)
    except Exception as e:
        logger.warning(
            "Standard loading failed (likely Keras version mismatch). "
            "Attempting to rebuild architecture and load weights only..."
        )
        logger.debug("Load error: %s", e)
        # Fallback: Rebuild the model and load only the weights
        model = build_model()
        model.load_weights(str(path))
        logger.info("Model weights loaded into rebuilt architecture.")

    return model
