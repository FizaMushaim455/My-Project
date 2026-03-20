"""
inference_pipeline.py — Real-time ASL recognition camera loop.

Converts the live-inference code from ``ASL_Real-Time.ipynb`` into a
reusable pipeline function.  Presses **Escape** to exit the loop; the
accumulated word string is returned for downstream processing (e.g. TTS).
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config.settings import (
    CAMERA_INDEX,
    CAPTURE_INTERVAL,
    IMG_SIZE,
    LABELS_DICT,
    MIN_VALUE,
    MODEL_PATH,
    ROI,
)
from src.models.model_utils import load_model
from src.utils.image_processing import preprocess_frame

logger = logging.getLogger(__name__)


def run_realtime(
    model_path: Path = MODEL_PATH,
    labels_dict: dict = LABELS_DICT,
    img_size: int = IMG_SIZE,
    min_value: int = MIN_VALUE,
    roi: tuple = ROI,
    camera_index: int = CAMERA_INDEX,
    capture_interval: int = CAPTURE_INTERVAL,
) -> str:
    """Open the webcam and perform real-time ASL gesture recognition.

    Displays two windows:
    - ``LIVE`` — the raw camera feed with ROI rectangle and overlaid text.
    - ``Processed`` — the thresholded binary image fed to the model.

    Controls:
    - **Esc**: Stop and synthesize speech.
    - **Backspace**: Undo last character.
    - **'C'**: Clear all text.

    Args:
        model_path: Path to the pre-trained ``.h5`` model file.
        labels_dict: Mapping from class index to ASL character.
        img_size: Image side length expected by the model.
        min_value: Otsu threshold minimum.
        roi: Region of interest ``(x1, y1, x2, y2)`` in camera frame pixels.
        camera_index: System camera device index.
        capture_interval: Number of frames between each letter capture.

    Returns:
        The full recognised text string (space-separated words).
    """
    model = load_model(model_path)
    logger.info("Starting real-time inference. Press Esc to stop.")

    source = cv2.VideoCapture(camera_index)
    if not source.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {camera_index}. "
            "Check your camera connection or set CAMERA_INDEX in .env."
        )

    x1, y1, x2, y2 = roi
    roi_color = (0, 255, 0)

    count: int = 0
    prev_char: str = " "
    recognised_string: str = ""
    start_time = time.time()

    try:
        while True:
            ret, frame = source.read()
            if not ret:
                logger.warning("Failed to read frame from camera; skipping.")
                continue

            # Calculate timer
            elapsed = int(time.time() - start_time)
            timer_text = time.strftime("%H:%M:%S", time.gmtime(elapsed))

            # Draw ROI rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)

            # Pre-process frame
            thresholded, model_input = preprocess_frame(frame, roi, img_size, min_value)

            # Predict every frame; capture letter every capture_interval frames
            result = model.predict(model_input, verbose=0)
            label_idx = int(np.argmax(result, axis=1)[0])

            count += 1

            # Update frame counter display (visual cue for next capture)
            progress = (count % 300) // 100  # legacy counter
            cv2.putText(
                frame,
                f"Next in: {capture_interval - count}",
                (x1, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
            )

            if count >= capture_interval:
                count = 0  # reset
                prev_char = labels_dict[label_idx]
                if label_idx == 0:
                    recognised_string += " "
                else:
                    recognised_string += prev_char
                logger.debug("Captured: '%s'", prev_char)

            # --- UI Overlays ---
            # Timer (Top Left)
            cv2.putText(frame, f"REC {timer_text}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Controls Help (Bottom Left)
            cv2.putText(frame, "Esc: Stop | Backspace: Undo | C: Clear", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Current Letter
            cv2.putText(
                frame,
                f"Sign: {labels_dict[label_idx]}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            
            # Recognised Sequence
            cv2.putText(
                frame,
                f"Text: {recognised_string}",
                (x2 + 20, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Processed", thresholded)
            cv2.imshow("LIVE", frame)

            key = cv2.waitKey(1)
            if key == 27:  # Escape
                logger.info("Escape pressed — stopping.")
                break
            elif key == 8:  # Backspace
                if len(recognised_string) > 0:
                    recognised_string = recognised_string[:-1]
                    logger.info("Undo: '%s'", recognised_string)
            elif key == ord('c') or key == ord('C'):
                recognised_string = ""
                logger.info("Text cleared.")

    finally:
        source.release()
        cv2.destroyAllWindows()
        logger.info("Camera released. Final string: '%s'", recognised_string.strip())

    return recognised_string.strip()
