"""
dataset_loader.py — Load and pre-process the ASL image dataset.

Expects the following directory layout (as used in ``ASL_train.ipynb``):

    DATASET/
        train/
            0/   A/   B/   ...   Z/
        test/
            0/   A/   B/   ...   Z/

Each class sub-folder contains grayscale hand-gesture images.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tensorflow.keras.utils import to_categorical  # type: ignore

from src.config.settings import DATASET_PATH, IMG_SIZE, MIN_VALUE
from src.utils.image_processing import preprocess_image_file

logger = logging.getLogger(__name__)


def _load_split(
    split_path: Path,
    img_size: int,
    min_value: int,
) -> Tuple[List[np.ndarray], List[int]]:
    """Load images from a single dataset split (train or test).

    Args:
        split_path: Path to the split directory (e.g. ``DATASET/train``).
        img_size: Target image size in pixels (square).
        min_value: Otsu threshold minimum.

    Returns:
        Tuple of ``(images, labels)`` where images are pre-processed arrays.
    """
    images: List[np.ndarray] = []
    labels: List[int] = []

    classes = sorted(os.listdir(split_path))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    logger.info("Found %d classes in '%s': %s", len(classes), split_path, classes)

    for class_name in classes:
        class_dir = split_path / class_name
        if not class_dir.is_dir():
            continue
        idx = class_to_idx[class_name]
        img_names = os.listdir(class_dir)

        for img_name in img_names:
            img_path = str(class_dir / img_name)
            try:
                processed = preprocess_image_file(img_path, img_size, min_value)
                images.append(processed)
                labels.append(idx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping '%s': %s", img_path, exc)

    logger.info(
        "Loaded %d images from '%s'.", len(images), split_path
    )
    return images, labels


def load_dataset(
    dataset_path: Path = DATASET_PATH,
    img_size: int = IMG_SIZE,
    min_value: int = MIN_VALUE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load, pre-process, and combine the full ASL dataset into arrays.

    Loads ``train/`` and ``test/`` splits, pre-processes every image, and
    returns normalised feature arrays with one-hot-encoded labels.

    Args:
        dataset_path: Root path of the dataset directory.
        img_size: Target image size in pixels (square).
        min_value: Otsu threshold minimum (see :mod:`src.utils.image_processing`).

    Returns:
        Tuple ``(X, y)`` where:
        - ``X`` shape: ``(N, img_size, img_size, 1)`` float32
        - ``y`` shape: ``(N, num_classes)`` float32 (one-hot)

    Raises:
        FileNotFoundError: If ``dataset_path`` does not exist.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_path}\n"
            "Please place your DATASET/ folder in the project root."
        )

    all_images: List[np.ndarray] = []
    all_labels: List[int] = []
    num_classes = 0

    for split in sorted(os.listdir(dataset_path)):
        split_path = dataset_path / split
        if not split_path.is_dir():
            continue
        imgs, lbls = _load_split(split_path, img_size, min_value)
        all_images.extend(imgs)
        all_labels.extend(lbls)
        num_classes = max(num_classes, max(lbls) + 1) if lbls else num_classes

    if not all_images:
        raise ValueError("No images found in the dataset. Check your DATASET/ structure.")

    X = np.array(all_images, dtype=np.float32) / 255.0
    X = np.reshape(X, (X.shape[0], img_size, img_size, 1))
    y = to_categorical(np.array(all_labels), num_classes=num_classes)

    logger.info(
        "Dataset loaded: X=%s  y=%s  num_classes=%d", X.shape, y.shape, num_classes
    )
    return X, y
