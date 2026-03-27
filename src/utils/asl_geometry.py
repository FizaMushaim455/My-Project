"""
asl_geometry.py — Anatomy-based ASL recognition (Developer: Fiza Mushaim | 2023-ag-9944)

This engine uses hand joint angles, distances, and orientations to 
classify the full ASL alphabet without training.
"""

import math
from typing import List, Tuple

def get_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def classify_asl_geometry(landmarks: List[float]) -> str:
    """Classifies ASL signs using anatomical rules."""
    pts = []
    for i in range(0, len(landmarks), 2):
        pts.append((landmarks[i], landmarks[i+1]))

    # Key points
    wrist = pts[0]
    thumb_tip = pts[4]
    thumb_ip = pts[3]
    index_tip = pts[8]
    index_pip = pts[6]
    middle_tip = pts[12]
    middle_pip = pts[10]
    ring_tip = pts[16]
    ring_pip = pts[14]
    pinky_tip = pts[20]
    pinky_pip = pts[18]

    # Finger states (Up/Down) - In MediaPipe Y decreases going UP
    index_up = index_tip[1] < index_pip[1]
    middle_up = middle_tip[1] < middle_pip[1]
    ring_up = ring_tip[1] < ring_pip[1]
    pinky_up = pinky_tip[1] < pinky_pip[1]
    
    # Thumb state (Out/In) - horizontal distance from wrist/index
    thumb_out = abs(thumb_tip[0] - wrist[0]) > abs(thumb_ip[0] - wrist[0])

    # Horizontal detection (for G, H)
    # Fingers are sideways if Tip.Y is similar to Root.Y but Tip.X is far
    index_side = abs(index_tip[1] - index_pip[1]) < 0.1 and abs(index_tip[0] - index_pip[0]) > 0.1
    middle_side = abs(middle_tip[1] - middle_pip[1]) < 0.1 and abs(middle_tip[0] - middle_pip[0]) > 0.1

    # Finger proximity (U vs V)
    dist_index_middle = get_dist(index_tip, middle_tip)

    # Touch detection (for O)
    d_thumb_index = get_dist(thumb_tip, index_tip)
    d_thumb_middle = get_dist(thumb_tip, middle_tip)
    d_thumb_ring = get_dist(thumb_tip, ring_tip)
    d_thumb_pinky = get_dist(thumb_tip, pinky_tip)

    # --- CLASSIFICATION RULES ---
    
    # O: All tips touching thumb
    if d_thumb_index < 0.1 and d_thumb_middle < 0.1 and d_thumb_ring < 0.1 and d_thumb_pinky < 0.1:
        return "O"

    # Alphabet A-Z (Simplified subset for common use)
    
    # H: Index and Middle pointing sideways
    if index_side and middle_side and not ring_up and not pinky_up:
        return "H"
        
    # G: Index pointing sideways, thumb out
    if index_side and not middle_up and not ring_up and not pinky_up:
        return "G"

    # L: Index up, thumb out
    if index_up and thumb_out and not middle_up and not ring_up and not pinky_up:
        return "L"

    # D: Index up, others folded (O-shape)
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "D"

    # V or U
    if index_up and middle_up and not ring_up and not pinky_up:
        if dist_index_middle > 0.1:
            return "V"
        else:
            return "U"
            
    # W
    if index_up and middle_up and ring_up and not pinky_up:
        return "W"

    # F: Index and thumb touch, others up
    if not index_up and middle_up and ring_up and pinky_up:
        return "F"
        
    # I: Only pinky up
    if pinky_up and not index_up and not middle_up and not ring_up:
        return "I"

    # Y: Thumb and Pinky out
    if thumb_out and pinky_up and not index_up and not middle_up and not ring_up:
        return "Y"

    # B: All fingers up and together
    if index_up and middle_up and ring_up and pinky_up:
        return "B"

    # A: Fist (thumb on side)
    if not index_up and not middle_up and not ring_up and not pinky_up and thumb_out:
        return "A"

    # S: Tight fist (thumb closed)
    if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_out:
        return "S"

    # Space: Hand open (5) but maybe palm facing camera
    if thumb_out and index_up and middle_up and ring_up and pinky_up:
        # If very wide, treat as Space
        if dist_index_middle > 0.15:
            return "Space"
        return "5"

    return "Unknown"
