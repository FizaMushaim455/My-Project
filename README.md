# ASL Sign-To-Speech Conversion

A real-time American Sign Language (ASL) gesture recognition system that converts hand signs into English text and provides audible **Urdu speech** output.

## Overview

This project uses a Deep Learning (CNN) model to recognize 27 ASL classes (letters A-Z and digit 0) from a live webcam feed. Once a sequence of signs is captured, it is translated into Urdu and spoken using a text-to-speech engine.

## Features

- **Real-time Detection**: Fast and accurate hand gesture recognition via webcam.
- **Urdu Translation**: Built-in mapping for common English words to Urdu.
- **Voice Output**: Audible speech synthesis in Urdu.
- **Modular Pipeline**: Clean separation of data, model, and application logic.

## Getting Started

### 1. Prerequisites

Ensure you have Python installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Application

Launch the real-time recognition tool:

```bash
python main.py run
```

- Position your hand inside the green square.
- The system will detect letters based on your gestures.
- A **Next in: X** countdown shows you when the current sign will be captured.
- A **session timer** (REC 00:00:00) is displayed in the top-left corner.

### Keyboard Controls

| Key | Action |
| --- | --- |
| **Esc** | Stop and synthesize Urdu speech |
| **Backspace** | **Undo** (remove the last captured letter) |
| **C** | **Clear** (delete all captured text) |

### 3. Training (Optional)

If you have a dataset, you can retrain the model:

```bash
python main.py train
```

## Project Logic

- **`src/`**: Contains the core Python modules for image processing, model architecture, and audio synthesis.
- **`main.py`**: The main entry point for the application.
- **`asl_classifier.h5`**: The pre-trained model weights.
