# AI Video Light

An intelligent lighting control system leveraging computer vision with a minimal hardware footprint.

[Demo](https://www.youtube.com/shorts/tINytdX_a-E)

## Overview

This project implements an AI-powered vision light system using:
- Raspberry Pi
- Raspberry Pi Camera
- Tapo WiFi light

The system employs a compact CNN+LSTM neural network architecture that can run efficiently on edge devices. The model was trained on manually collected data to recognize lighting patterns and automate control.

## How It Works

The system continuously captures video frames and analyzes them to make lighting decisions:

1. **Video Capture**: The Raspberry Pi camera captures frames at a specified FPS.
2. **Frame Analysis**: The CNN+LSTM model processes sequences of frames to predict optimal lighting conditions.
3. **Light Control**: Based on predictions, the system automatically toggles the Tapo WiFi light.
4. **Data Collection**: The system records user interactions to potentially improve future performance.

## Key Components

- **Camera Module**: Manages video capture from the Raspberry Pi camera.
- **Light Control**: Interfaces with the Tapo WiFi light for remote operation.
- **Information Store**: Maintains a buffer of recent frames and light states.
- **AI Model**: A space-efficient CNN+LSTM architecture for video sequence analysis.

## Usage

Run the system with default settings (AI control enabled):
```bash
python main.py
```

Run in monitoring mode without AI control:
```bash
python main.py --no-ai
```

## Technical Details

- The system maintains a configurable buffer of recent video frames
- Frame sequences are normalized using precalculated mean and standard deviation values
- The model outputs probability scores that are thresholded to make binary lighting decisions
- When operating without AI, the system can record lighting change events for future training

## Implementation Notes

The main execution loop:
1. Captures frames from the camera at the specified FPS
2. Retrieves current light status
3. Records frame and light information to the buffer
4. Either:
   - Monitors and saves lighting changes (in no-ai mode), or
   - Processes frame sequences to predict and automatically adjust lighting
