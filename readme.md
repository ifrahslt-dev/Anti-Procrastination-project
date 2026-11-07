
# Computer Vision Dashboard

A real-time computer vision application built with Python, OpenCV, and YOLO26.

## Features

- Real-time webcam object detection
- AI-powered object recognition
- Cellphone detection
- MP3 alarm when a cellphone is detected
- Live FPS counter
- Object counting
- Adjustable detection confidence
- Screenshot capture

## Technologies

- Python
- OpenCV
- YOLO26 (Ultralytics)
- Pygame
- NumPy

## Requirements

- Python 3.10 or newer
- A webcam
- macOS, Windows, or Linux

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Enter the project folder:

```bash
cd computer-vision
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

On Windows, use:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Add your own `alarm.mp3` file to the project folder if one is not included.

## Run the application

```bash
python dashboard.py
```

The YOLO model will download automatically on its first run.

## Keyboard Controls

| Key | Action |
|---|---|
| Q | Quit |
| S | Save screenshot |
| + | Increase confidence |
| - | Decrease confidence |

## Future Improvements

- Improved cellphone detection
- Custom-trained object detection
- More alert sounds
- Detection history
- Performance optimization

## Author

Created as a Computer Science portfolio project.
