"""
hand_tracking.py — MediaPipe Tasks Hand Landmarks extraction utility.

Updated for modern MediaPipe (0.10+) tasks API.
"""

import logging
import os
from pathlib import Path
from typing import Optional, List

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

logger = logging.getLogger(__name__)

# Path to the task model download
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

class HandTracker:
    """Wraps MediaPipe Tasks HandLandmarker to extract joints."""

    def __init__(self, min_conf: float = 0.5):
        """Initialize the HandLandmarker."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"MediaPipe task model not found at {MODEL_PATH}")

        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Skeleton connections for manual drawing
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),           # Index
            (9, 10), (10, 11), (11, 12),              # Middle
            (13, 14), (14, 15), (15, 16),             # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
            (5, 9), (9, 13), (13, 17)                 # Palm
        ]

    def get_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract 21 normalized landmarks (x, y) relative to wrist."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect
        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        # Process first hand detected
        landmarks = result.hand_landmarks[0]
        
        data_x = [lm.x for lm in landmarks]
        data_y = [lm.y for lm in landmarks]

        # Normalized features relative to wrist (index 0)
        base_x, base_y = data_x[0], data_y[0]
        norm_x = [x - base_x for x in data_x]
        norm_y = [y - base_y for y in data_y]

        # Scale normalization (square bounding box)
        max_dist = max(max(map(abs, norm_x)), max(map(abs, norm_y)))
        if max_dist > 0:
            norm_x = [x / max_dist for x in norm_x]
            norm_y = [y / max_dist for y in norm_y]

        features = []
        for x, y in zip(norm_x, norm_y):
            features.extend([x, y])

        return np.array(features, dtype=np.float32)

    def draw_skeleton(self, frame: np.ndarray) -> np.ndarray:
        """Draw the hand digital skeleton using the detected landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return frame

        height, width = frame.shape[:2]
        for hand_landmarks in result.hand_landmarks:
            # Draw joints
            for lm in hand_landmarks:
                px, py = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

            # Draw bones
            for start_idx, end_idx in self.connections:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                p1 = (int(start.x * width), int(start.y * height))
                p2 = (int(end.x * width), int(end.y * height))
                cv2.line(frame, p1, p2, (255, 255, 255), 2)
                
        return frame
