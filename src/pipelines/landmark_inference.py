"""
landmark_inference.py — High-accuracy real-time recognition using MediaPipe.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.config.settings import (
    CAMERA_INDEX,
    LABELS_DICT,
    ROI,
    STABILITY_THRESHOLD,
)
from src.utils.hand_tracking import HandTracker
from src.utils.asl_geometry import classify_asl_geometry

logger = logging.getLogger(__name__)


def run_landmark_inference(
    model_path: Path = None, # No longer strictly needed for geometric mode
    labels_dict: dict = LABELS_DICT,
    roi: tuple = ROI,
    camera_index: int = CAMERA_INDEX,
    stability_threshold: int = STABILITY_THRESHOLD,
) -> str:
    """Perform real-time recognition using geometric rules (Perfect Stability)."""
    
    tracker = HandTracker(min_conf=0.7)
    
    source = cv2.VideoCapture(camera_index)
    recognised_string: str = ""
    stability_buffer: list[str] = []
    last_added_sign: Optional[str] = None # Guard against duplicates
    
    start_time = time.time()
    logger.info("MediaPipe GEOMETRIC (No-Train) Inference started.")

    # --- FULL SCREEN WINDOW SETUP ---
    window_name = "ASL PRO Tool"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ret, frame = source.read()
            if not ret: break

            # Upscale for better full-screen quality (1280x720)
            frame = cv2.resize(frame, (1280, 720))
            
            # UI Timer
            elapsed = int(time.time() - start_time)
            timer_text = time.strftime("%H:%M:%S", time.gmtime(elapsed))

            # Landmark Detection
            landmarks = tracker.get_landmarks(frame)
            frame = tracker.draw_skeleton(frame)
            
            current_sign = "None"
            if landmarks is not None:
                # USE GEOMETRY ENGINE
                current_sign = classify_asl_geometry(landmarks.tolist())

                # Stability Logic
                if not stability_buffer or stability_buffer[-1] == current_sign:
                    stability_buffer.append(current_sign)
                else:
                    stability_buffer = [current_sign]

                # --- NEW DEBOUNCING LOGIC ---
                if len(stability_buffer) >= stability_threshold and current_sign != "Unknown":
                    # Only add if it's a NEW sign (or hand was reset)
                    if current_sign != last_added_sign:
                        if current_sign == "Space":
                            if not recognised_string.endswith(" "):
                                recognised_string += " "
                        else:
                            recognised_string += current_sign
                        
                        last_added_sign = current_sign # Mark as added
                    
                    stability_buffer = []
                    cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 10)
            else:
                stability_buffer = []
                last_added_sign = None # Reset when hand is removed, allowing same sign again

            # --- PREMIUM UI RENDERING ---
            overlay = frame.copy()
            
            # Top Bar (Semi-transparent)
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (30, 30, 30), -1)
            # Bottom Bar (Semi-transparent)
            cv2.rectangle(overlay, (0, frame.shape[0] - 100), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
            
            # Blend overlay
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # 1. Mode & Timer (Top Left)
            cv2.putText(frame, "PRO GEOMETRIC ENGINE", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"LIVE REC: {timer_text}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 2. Detected Sign (Top Center)
            sign_color = (0, 255, 255) if current_sign != "None" else (150, 150, 150)
            cv2.putText(frame, f"SIGN: {current_sign}", (frame.shape[1]//2 - 120, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, sign_color, 4, cv2.LINE_AA)

            # 3. Sentence (Bottom Left)
            cv2.putText(frame, "CUMULATIVE SENTENCE:", (15, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, recognised_string, (15, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 4. Progress Bar (Sleek)
            stability_ratio = min(len(stability_buffer) / stability_threshold, 1.0)
            bar_width = int(frame.shape[1] * stability_ratio)
            cv2.rectangle(frame, (0, 80), (bar_width, 85), (0, 255, 0), -1)

            # 5. Controls (Bottom Right - Small)
            cv2.putText(frame, "[Esc] Finish | [Bksp] Undo | [C] Clear", (frame.shape[1] - 320, frame.shape[0] - 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            
            # 6. User Branding (Bottom Right - Prominent)
            cv2.putText(frame, "Fiza Mushaim", (frame.shape[1] - 320, frame.shape[0] - 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "2023-ag-9944", (frame.shape[1] - 320, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1)
            if key == 27: break
            elif key == 8 and recognised_string: recognised_string = recognised_string[:-1]
            elif (key == ord('c') or key == ord('C')): recognised_string = ""

    finally:
        source.release()
        cv2.destroyAllWindows()
        
    return recognised_string.strip()
