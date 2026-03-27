"""
settings.py — Project Configuration (Developer: Fiza Mushaim | 2023-ag-9944)
"""

from pathlib import Path

# --- CORE PATHS ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# --- RECOGNITION PARAMETERS ---
CAMERA_INDEX = 0
ROI = (100, 100, 300, 300) # (y1, x1, y2, x2)

# --- SPEECH PARAMETERS ---
AUDIO_OUTPUT_FILE = "output_speech.mp3"
TTS_LANGUAGE = "ur"
TTS_SLOW = False

# Stability threshold (number of consecutive frames to capture a letter)
STABILITY_THRESHOLD = 12

# Color Enhancement
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# --- DICTIONARIES ---
ENGLISH_TO_URDU_DICT = {
    "hello": "ہیلو",
    "how": "کیسے",
    "are": "ہیں",
    "you": "آپ",
    "i": "میں",
    "am": "ہوں",
    "fine": "ٹھیک",
    "thanks": "شکریہ",
    "good": "اچھا",
    "morning": "صبح",
    "night": "رات",
    "please": "براہ مہربانی",
    "sorry": "معذرت",
    "yes": "جی ہاں",
    "no": "نہیں",
    "space": " ",
}

LABELS_DICT = {
    0: "Space", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 
    8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M", 14: "N", 15: "O", 
    16: "P", 17: "Q", 18: "R", 19: "S", 20: "T", 21: "U", 22: "V", 23: "W", 
    24: "X", 25: "Y", 26: "Z"
}

NUM_CLASSES = len(LABELS_DICT)
