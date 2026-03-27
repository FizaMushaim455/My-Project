"""
landmark_model.py — MLP Classifier for hand landmark coordinates.

Input: 42-dimensional vector (21 hand joints * [x, y]).
Output: 27 classes (0, A–Z).
"""

import logging
from tensorflow.keras.layers import Dense, Dropout, Input  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore

from src.config.settings import NUM_CLASSES

logger = logging.getLogger(__name__)


def build_landmark_model(num_classes: int = NUM_CLASSES) -> Sequential:
    """Build a Multi-Layer Perceptron for landmark-based classification.

    Args:
        num_classes: Number of ASL classes (default 27).

    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential(
        [
            Input(shape=(42,)),  # 21 landmarks * 2 (x,y)
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Landmark model built and compiled successfully.")
    return model
