"""
image_processing.py — Shared OpenCV pre-processing helpers.

These functions encapsulate the exact pipeline used in both training
(``ASL_train.ipynb``) and real-time inference (``ASL_Real-Time.ipynb``):

    BGR → Grayscale → GaussianBlur → AdaptiveThreshold → Otsu → Resize → Normalise
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_frame(
    frame: np.ndarray,
    roi: Tuple[int, int, int, int],
    img_size: int,
    min_value: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-process a raw BGR camera frame for model inference.

    1. Crops the region of interest (ROI).
    2. Converts to grayscale.
    3. Applies Gaussian blur.
    4. Applies adaptive + Otsu thresholding.
    5. Resizes to ``img_size × img_size``.
    6. Normalises pixel values to [0, 1].
    7. Reshapes to ``(1, img_size, img_size, 1)`` ready for the model.

    Args:
        frame: Raw BGR image from ``cv2.VideoCapture.read()``.
        roi: ``(x1, y1, x2, y2)`` crop coordinates.
        img_size: Target square size (pixels) after resize.
        min_value: Threshold minimum for Otsu binarisation.

    Returns:
        A tuple of ``(processed_roi, model_input)`` where:
        - ``processed_roi`` is the thresholded grayscale crop (uint8, H×W).
        - ``model_input`` is normalised and reshaped (float64, 1×H×W×1).
    """
    x1, y1, x2, y2 = roi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop = gray[y1:y2, x1:x2]

    # Enhanced Contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(crop)

    blurred = cv2.GaussianBlur(enhanced, (5, 5), 2)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )
    _, thresholded = cv2.threshold(
        adaptive,
        min_value,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    resized = cv2.resize(thresholded, (img_size, img_size))
    normalised = resized / 255.0
    model_input = np.reshape(normalised, (1, img_size, img_size, 1))

    return thresholded, model_input


def preprocess_image_file(
    image_path: str,
    img_size: int,
    min_value: int,
) -> np.ndarray:
    """Pre-process a static image file for model training or evaluation.

    Applies the same pipeline as :func:`preprocess_frame` but reads from disk
    and returns only the resized, normalised array (no model-input reshaping).

    Args:
        image_path: Absolute or relative path to the image file.
        img_size: Target square size (pixels) after resize.
        min_value: Threshold minimum for Otsu binarisation.

    Returns:
        Normalised grayscale array of shape ``(img_size, img_size)``, or
        ``None`` if the image cannot be read.

    Raises:
        ValueError: If the image cannot be read from ``image_path``.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )
    _, thresholded = cv2.threshold(
        adaptive,
        min_value,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    resized = cv2.resize(thresholded, (img_size, img_size))
    return resized
