# 🖐️ ASL Sign-To-Speech Conversion (Geometric Edition)

A high-precision, real-time American Sign Language (ASL) recognition system that converts hand gestures into typed text and audible **Urdu speech**.

---

## 👨‍💻 Developed By: **Fiza Mushaim**
## 📜 AG Number: **2023-ag-9944**

---

## 🚀 Step-by-Step Setup Guide

Follow these steps to get the project running on your machine:

### 1. Prerequisites
Ensure you have **Python 3.10+** installed. This project is optimized for Python 3.13.

### 2. Install Dependencies
Open your terminal inside the project folder (`My-Project`) and run:
```bash
pip install mediapipe opencv-python tensorflow gTTS playsound
```
*(Note: MediaPipe is used for the high-end 3D joint tracking)*

### 3. Launch the Application
Run the main recognition script:
```bash
python main.py run
```
---

## 💎 Features & Capabilities

-   **PRO Geometric Engine**: No training required! It uses 3D hand anatomy to recognize signs instantly.
-   **Steady & Robust**: Works perfectly even on plain white backgrounds and varying light.
-   **Digital Skeleton**: Displays a real-time green skeleton over your hand joints.
-   **Urdu Voice Output**: Converts the recognized English letters into spoken Urdu.

---

## ⌨️ Controls & Usage

| Key | Action |
| --- | --- |
| **Esc** | Stop the session and speak out in Urdu. |
| **Backspace** | **Undo** (remove the last captured character). |
| **C** | **Clear** (delete all captured text). |

### 📚 Learning the Signs
For the best results, refer to our project-specific guide:
👉 **[ASL_CHEAT_SHEET.md](ASL_CHEAT_SHEET.md)**

---

## 🔗 Visual Reference Links
- [Stable ASL Alphabet Chart (Wikipedia)](https://upload.wikimedia.org/wikipedia/commons/e/e3/American_Sign_Language_Alphabet.svg)
- [Official ASDC ASL Guide](https://www.deafchildren.org/resources/asl-alphabet/)
