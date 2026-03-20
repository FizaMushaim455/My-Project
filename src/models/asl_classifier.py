"""
asl_classifier.py — CNN model definition for ASL hand gesture recognition.

Reproduces the exact architecture from ``ASL_train.ipynb`` as a reusable
``build_model()`` factory function, eliminating global state.
"""

import logging
import os

from tensorflow.keras.layers import (  # type: ignore
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential  # type: ignore

from src.config.settings import (
    CUDA_VISIBLE_DEVICES,
    DROPOUT_RATE,
    IMG_SIZE,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)


def build_model(
    img_size: int = IMG_SIZE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
) -> Sequential:
    """Build and compile the ASL CNN classifier.

    Architecture (matches original notebook exactly):
        - Input(128×128×1)
        - Conv2D(32, 3×3, relu) → MaxPool(2×2)
        - Conv2D(32, 3×3, relu) → MaxPool(2×2)
        - Flatten
        - Dense(128, relu) → Dropout
        - Dense(96,  relu) → Dropout
        - Dense(64,  relu)
        - Dense(num_classes, softmax)

    Compiled with Adam optimiser and categorical cross-entropy loss.

    Args:
        img_size: Height and width of the input images (square).
        num_classes: Number of output classes (27 for 0 + A–Z).
        dropout_rate: Dropout fraction applied after the first two dense layers.

    Returns:
        A compiled Keras ``Sequential`` model.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    model = Sequential(
        [
            Input(shape=(img_size, img_size, 1)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(dropout_rate),
            Dense(96, activation="relu"),
            Dropout(dropout_rate),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Model built successfully.")
    model.summary(print_fn=logger.debug)
    return model
