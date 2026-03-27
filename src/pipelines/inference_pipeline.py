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
    CLAHE_CLIP_LIMIT,
    CLAHE_GRID_SIZE,
    IMG_SIZE,
    LABELS_DICT,
    MIN_VALUE,
    MODEL_PATH,
    ROI,
    STABILITY_THRESHOLD,
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
    stability_threshold: int = STABILITY_THRESHOLD,
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

    # Stability & Capture Logic
    recognised_string: str = ""
    stability_buffer: list[str] = []
    stability_threshold: int = 8  # Frames of consistent detection needed
    
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

            # Pre-process frame
            thresholded, model_input = preprocess_frame(frame, roi, img_size, min_value)

            # Predict current frame
            result = model.predict(model_input, verbose=0)
            label_idx = int(np.argmax(result, axis=1)[0])
            current_sign = labels_dict[label_idx]

            # Update stability buffer
            if not stability_buffer or stability_buffer[-1] == current_sign:
                stability_buffer.append(current_sign)
            else:
                stability_buffer = [current_sign]  # Reset if sign changes

            # Calculate stability progress (0.0 to 1.0)
            stability_ratio = min(len(stability_buffer) / stability_threshold, 1.0)

            # Draw ROI and overlays
            cv2.rectangle(frame, (x1, y1), (x2, y2), roi_color, 2)
            
            # Draw Stability Bar (under ROI)
            bar_width = x2 - x1
            bar_height = 10
            cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 5 + bar_height), (50, 50, 50), -1)
            cv2.rectangle(frame, (x1, y2 + 5), (x1 + int(bar_width * stability_ratio), y2 + 5 + bar_height), (0, 255, 0), -1)

            # Capture logic: when threshold is reached
            if len(stability_buffer) >= stability_threshold:
                # Capture the letter
                if label_idx == 0:  # Code for space
                    if not recognised_string.endswith(" "):
                        recognised_string += " "
                else:
                    recognised_string += current_sign
                
                logger.debug("Locked in: '%s' | String: '%s'", current_sign, recognised_string)
                
                # Clear buffer to wait for the next sign (prevent multiple captures of same hold)
                # Usually we want a "cooldown" or wait for a change
                stability_buffer = [] 
                # Optional: display a flash or sound?
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 5)

            # --- UI Overlays ---
            # Timer (Top Left)
            cv2.putText(frame, f"REC {timer_text}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Controls Help (Bottom Left)
            cv2.putText(frame, "Esc: Stop | Backspace: Undo | C: Clear", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Current Guess (Floaty)
            cv2.putText(
                frame,
                f"Detecting: {current_sign}",
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            
            # Recognised Sequence
            cv2.putText(
                frame,
                f"Sentence: {recognised_string}",
                (x1, y2 + 45),
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
