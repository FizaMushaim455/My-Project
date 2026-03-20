"""
train_pipeline.py — End-to-end training pipeline for the ASL classifier.

Converts the training notebook (``ASL_train.ipynb``) into a reusable,
parameterised pipeline function with checkpointing and evaluation.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore

from src.config.settings import (
    DATASET_PATH,
    EPOCHS,
    IMG_SIZE,
    MIN_VALUE,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SPLIT,
)
from src.data.dataset_loader import load_dataset
from src.models.asl_classifier import build_model
from src.models.model_utils import save_model

logger = logging.getLogger(__name__)


def run_training(
    dataset_path: Path = DATASET_PATH,
    model_output_path: Path = MODEL_PATH,
    img_size: int = IMG_SIZE,
    min_value: int = MIN_VALUE,
    epochs: int = EPOCHS,
    validation_split: float = VALIDATION_SPLIT,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    plot_output_dir: Optional[Path] = None,
) -> None:
    """Load data, train the ASL CNN, evaluate, and save the model.

    Args:
        dataset_path: Root path to the DATASET/ directory.
        model_output_path: Where to write the trained ``.h5`` model file.
        img_size: Image side length (pixels).
        min_value: Otsu threshold minimum.
        epochs: Number of training epochs.
        validation_split: Fraction of training data used for validation.
        test_size: Fraction of the full dataset held out for final evaluation.
        random_state: Random seed for reproducibility.
        plot_output_dir: Optional directory for saving training history plots.
            If ``None``, plots are not saved.
    """
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    logger.info("Loading dataset from '%s' …", dataset_path)
    X, y = load_dataset(dataset_path, img_size, min_value)

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Split: train=%d  test=%d", len(X_train), len(X_test)
    )

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    num_classes = y.shape[1]
    model = build_model(img_size=img_size, num_classes=num_classes)

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    checkpoint = ModelCheckpoint(
        "model-{epoch:03d}.h5",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        mode="auto",
    )

    logger.info("Starting training for %d epochs …", epochs)
    history = model.fit(
        X_train,
        y_train,
        shuffle=True,
        epochs=epochs,
        callbacks=[checkpoint],
        validation_split=validation_split,
    )

    # ------------------------------------------------------------------
    # 5. Evaluate on held-out test set
    # ------------------------------------------------------------------
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test loss: %.4f  |  Test accuracy: %.4f", test_loss, test_acc)

    # ------------------------------------------------------------------
    # 6. Save model
    # ------------------------------------------------------------------
    save_model(model, model_output_path)

    # ------------------------------------------------------------------
    # 7. Optional: save training history plots
    # ------------------------------------------------------------------
    if plot_output_dir is not None:
        plot_output_dir = Path(plot_output_dir)
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        _save_training_plots(history, plot_output_dir)


def _save_training_plots(history, output_dir: Path) -> None:
    """Save loss and accuracy curve plots as PNG files.

    Args:
        history: Keras ``History`` object returned by ``model.fit()``.
        output_dir: Directory where PNG files are written.
    """
    epochs_range = np.arange(len(history.history["loss"]))

    # Loss curve
    plt.figure()
    plt.plot(epochs_range, history.history["loss"], label="train_loss")
    plt.plot(epochs_range, history.history["val_loss"], label="val_loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()
    logger.info("Loss plot saved.")

    # Accuracy curve
    plt.figure()
    plt.plot(epochs_range, history.history["accuracy"], label="train_accuracy")
    plt.plot(epochs_range, history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(output_dir / "accuracy_curve.png")
    plt.close()
    logger.info("Accuracy plot saved.")
