"""
landmark_pipeline.py — End-to-end landmark-based training pipeline.

Features:
1. data_collection() — Records hand landmarks from webcam for a specific label.
2. train_landmark_model() — Trains the MLP on landmarks.csv.
"""

import logging
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

from src.utils.hand_tracking import HandTracker
from src.models.landmark_model import build_landmark_model
from src.config.settings import (
    CAMERA_INDEX, 
    LANDMARK_MODEL_PATH, 
    LANDMARKS_CSV, 
    LABELS_DICT,
    NUM_CLASSES
)

logger = logging.getLogger(__name__)

def collect_landmarks(label: str, num_samples: int = 100):
    """Open webcam and record landmarks for a specific character."""
    tracker = HandTracker(static_mode=False, min_conf=0.7)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    data = []
    logger.info("Starting collection for label: %s. Need %d samples.", label, num_samples)
    
    while len(data) < num_samples:
        ret, frame = cap.read()
        if not ret: break
        
        # UI
        text = f"Collecting '{label}': {len(data)}/{num_samples}"
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = tracker.draw_skeleton(frame)
        
        landmarks = tracker.get_landmarks(frame)
        if landmarks is not None:
            data.append(landmarks)
            time.sleep(0.05) # Small delay to get diverse frames
            
        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) == 27: break
        
    cap.release()
    cv2.destroyAllWindows()
    
    if data:
        df_new = pd.DataFrame(data)
        df_new.insert(0, 'label', label)
        
        if LANDMARKS_CSV.exists():
            df_old = pd.read_csv(LANDMARKS_CSV)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_final = df_new
            
        df_final.to_csv(LANDMARKS_CSV, index=False)
        logger.info("Saved %d samples for '%s' to %s", len(data), label, LANDMARKS_CSV)

def train_landmark_model():
    """Train the MLP on the accumulated landmarks CSV."""
    if not LANDMARKS_CSV.exists():
        logger.error("No training data found at %s", LANDMARKS_CSV)
        return
    
    df = pd.read_csv(LANDMARKS_CSV)
    
    # Map labels to indices
    inv_labels = {v: k for k, v in LABELS_DICT.items()}
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].map(inv_labels).values
    
    y_cat = to_categorical(y, num_classes=NUM_CLASSES)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)
    
    model = build_landmark_model(num_classes=NUM_CLASSES)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    model.save(LANDMARK_MODEL_PATH)
    logger.info("Landmark model saved to %s", LANDMARK_MODEL_PATH)
