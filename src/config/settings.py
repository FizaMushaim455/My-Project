"""
settings.py — Central configuration for the ASL Sign-To-Speech project.

All constants, paths, and hyperparameters live here.
Import from this module instead of hardcoding values anywhere else.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the project (two levels up from this file: src/config/ → root)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Pre-trained model file
MODEL_PATH: Path = PROJECT_ROOT / "asl_classifier.h5"

# Dataset directory (must contain train/ and test/ sub-folders)
DATASET_PATH: Path = PROJECT_ROOT / "DATASET"

# Audio output file used by gTTS
AUDIO_OUTPUT_FILE: str = str(PROJECT_ROOT / "output_speech.mp3")

# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

#: Side length (pixels) images are resized to before feeding the model.
IMG_SIZE: int = 128

#: Minimum threshold value for Otsu binarisation step.
MIN_VALUE: int = 70

#: Region of interest coordinates in the live camera frame: (x1, y1, x2, y2)
ROI: tuple[int, int, int, int] = (24, 24, 250, 250)

# ---------------------------------------------------------------------------
# Label mapping  (index → ASL character)
# ---------------------------------------------------------------------------

LABELS_DICT: dict[int, str] = {
    0: "0",
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
    10: "J",
    11: "K",
    12: "L",
    13: "M",
    14: "N",
    15: "O",
    16: "P",
    17: "Q",
    18: "R",
    19: "S",
    20: "T",
    21: "U",
    22: "V",
    23: "W",
    24: "X",
    25: "Y",
    26: "Z",
}

NUM_CLASSES: int = len(LABELS_DICT)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

EPOCHS: int = 20
VALIDATION_SPLIT: float = 0.3
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
DROPOUT_RATE: float = 0.40

# GPU configuration (empty string = use all available GPUs)
CUDA_VISIBLE_DEVICES: str = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# ---------------------------------------------------------------------------
# Real-time inference
# ---------------------------------------------------------------------------

#: Camera device index (0 = default webcam).
CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))

#: Number of frames between each letter capture.
CAPTURE_INTERVAL: int = 500

# ---------------------------------------------------------------------------
# Translation  (English word → Urdu word)
# ---------------------------------------------------------------------------

ENGLISH_TO_URDU_DICT: dict[str, str] = {
    "hello": "ہیلو",
    "how": "کیسے",
    "are": "ہیں",
    "you": "آپ",
    "hi": "سلام",
    "good": "اچھا",
    "morning": "صبح بخیر",
    "bye": "خدا حافظ",
    "thanks": "شکریہ",
    "yes": "ہاں",
    "no": "نہیں",
}

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

TTS_LANGUAGE: str = "ur"
TTS_SLOW: bool = False
