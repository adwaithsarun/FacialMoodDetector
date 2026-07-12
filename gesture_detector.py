
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import math
import os
import time
import urllib.request


IMAGE_FOLDER = "images"

IMAGE_PATHS = {
    "happy": os.path.join(IMAGE_FOLDER, "happy.png"),
    "sad": os.path.join(IMAGE_FOLDER, "sad.png"),
    "angry": os.path.join(IMAGE_FOLDER, "angry.png"),
}


DISPLAY_SIZE = (330, 330)

gesture_images = {}
for label, path in IMAGE_PATHS.items():
    if not os.path.exists(path):
        print(f"[WARNING] Could not find {path}. "
              f"Put an image there named '{label}.png'.")
        continue
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # keep alpha channel if present
    img = cv2.resize(img, DISPLAY_SIZE)
    gesture_images[label] = img


def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]

    
    if x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    if overlay.shape[2] == 4:  
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y + h, x:x + w, c] = (
                alpha * overlay[:, :, c] +
                (1 - alpha) * background[y:y + h, x:x + w, c]
            )
    else:  
        background[y:y + h, x:x + w] = overlay

    return background



MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading face landmark model (first run only, ~4MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
landmarker_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
face_landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)


LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14
LEFT_EYEBROW = 105
LEFT_EYE_TOP = 159
RIGHT_EYEBROW = 334
RIGHT_EYE_TOP = 386


def get_point(landmarks, index, w, h):
    lm = landmarks[index]
    return np.array([lm.x * w, lm.y * h])


def distance(p1, p2):
    return math.dist(p1, p2)



# ---------------------------------------------------------------
SMILE_THRESHOLD = 0.03       # mouth corners must rise above this to count as happy
SAD_THRESHOLD = -0.015       # mouth corners must droop below this to count as sad
NEUTRAL_SMILE_BAND = 0.02    # |smile_ratio| below this = mouth looks neutral
BROW_NEUTRAL_THRESHOLD = 0.2  # eyebrows above this distance = relaxed (not furrowed)
BROW_ANGRY_THRESHOLD = 0.19   # eyebrows below this distance = furrowed -> angry


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check your camera permissions.")

print("Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # mirror image, feels more natural
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap the frame in MediaPipe's image format and run detection.
    # VIDEO mode requires an increasing timestamp (milliseconds) each call.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

    current_gesture = "neutral"

    if results.face_landmarks:
        landmarks = results.face_landmarks[0]  # list of landmarks for the first face

        # Key points
        left_eye = get_point(landmarks, LEFT_EYE_OUTER, w, h)
        right_eye = get_point(landmarks, RIGHT_EYE_OUTER, w, h)
        mouth_left = get_point(landmarks, MOUTH_LEFT, w, h)
        mouth_right = get_point(landmarks, MOUTH_RIGHT, w, h)
        lip_top = get_point(landmarks, UPPER_LIP_TOP, w, h)
        lip_bottom = get_point(landmarks, LOWER_LIP_BOTTOM, w, h)
        left_brow = get_point(landmarks, LEFT_EYEBROW, w, h)
        left_eye_top = get_point(landmarks, LEFT_EYE_TOP, w, h)
        right_brow = get_point(landmarks, RIGHT_EYEBROW, w, h)
        right_eye_top = get_point(landmarks, RIGHT_EYE_TOP, w, h)

        # Normalize every measurement by the distance between the eyes,
        # so the numbers stay consistent regardless of distance to camera
        face_scale = distance(left_eye, right_eye)

        mouth_center_y = (lip_top[1] + lip_bottom[1]) / 2
        corner_avg_y = (mouth_left[1] + mouth_right[1]) / 2

        # Positive = corners raised above mouth center (smiling)
        # Negative = corners drooping below mouth center (frowning)
        smile_ratio = (mouth_center_y - corner_avg_y) / face_scale

        brow_left_dist = distance(left_brow, left_eye_top) / face_scale
        brow_right_dist = distance(right_brow, right_eye_top) / face_scale
        brow_avg = (brow_left_dist + brow_right_dist) / 2

        # --- Classification logic ---
        if smile_ratio > SMILE_THRESHOLD:
            current_gesture = "happy"
        elif smile_ratio < SAD_THRESHOLD:
            current_gesture = "sad"
        elif abs(smile_ratio) < NEUTRAL_SMILE_BAND and brow_avg > BROW_NEUTRAL_THRESHOLD:
            current_gesture = "neutral"
        elif brow_avg < BROW_ANGRY_THRESHOLD:
            current_gesture = "angry"
        else:
            current_gesture = "neutral"

        # Debug readout so you can calibrate the thresholds above
        cv2.putText(frame, f"smile_ratio: {smile_ratio:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"brow_avg: {brow_avg:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show detected gesture label
    cv2.putText(frame, f"Gesture: {current_gesture.upper()}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Overlay the matching image, top-right corner
    if current_gesture in gesture_images:
        img = gesture_images[current_gesture]
        x = w - DISPLAY_SIZE[0] - 20
        y = 20
        frame = overlay_image(frame, img, x, y)

    cv2.imshow("Face Gesture Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()