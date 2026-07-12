# FacialMoodDetector 😄😢😠

Real-time facial expression detector built with **MediaPipe**, **OpenCV**, and **NumPy**. It detects your facial expression through a webcam feed and displays the matching emoji live — happy, sad, or angry.

## Features

- Real-time face landmark detection using MediaPipe
- Classifies expressions into three moods: **Happy**, **Sad**, **Angry**
- Displays a corresponding emoji overlay on the webcam feed
- Lightweight — no deep learning model training required

## Tech Stack

- Python
- [MediaPipe](https://google.github.io/mediapipe/) — face landmark detection
- [OpenCV](https://opencv.org/) — video capture and image processing
- [NumPy](https://numpy.org/) — numerical operations on landmark coordinates

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/adwaithsarun/FacialMoodDetector.git
   cd FacialMoodDetector
   ```

2. (Optional but recommended) Create a virtual environment
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate # On macOS/Linux
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:

```bash
python main.py
```

- Make sure your webcam is connected and accessible.
- Press `q` to quit the application.

## How It Works

1. **Face Landmark Detection** — MediaPipe's Face Mesh model detects facial landmarks in real time.
2. **Feature Extraction** — Key landmark distances/ratios (e.g., mouth curvature, eyebrow position) are calculated using NumPy.
3. **Expression Classification** — A rule-based logic classifies the extracted features into Happy, Sad, or Angry.
4. **Emoji Overlay** — OpenCV overlays the matching emoji image onto the video feed in real time.

## Project Structure

```
FacialMoodDetector/
├── main.py
├── requirements.txt
├── emojis/
│   ├── happy.png
│   ├── sad.png
│   └── angry.png
├── .gitignore
└── README.md
```

## Future Improvements

- Add more emotion categories (surprised, neutral, fearful)
- Replace rule-based classification with a trained ML model
- Add a GUI for easier interaction

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

**Adwaithsarun**
