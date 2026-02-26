# Flay - Python OpenCV Camera Filter Application

## Cursor Cloud specific instructions

### Overview
Flay is a standalone Python desktop application that applies real-time face filters (cat ears, bear ears, fisheye, overlays) and background effects (blur, image replacement) to a webcam feed using PyQt5, OpenCV, dlib, and MediaPipe.

### Running the application
- Entry point: `python3 main.py`
- Requires a display server. In cloud/headless environments, start Xvfb first: `Xvfb :99 -screen 0 1280x720x24 &` then `export DISPLAY=:99`
- The app hardcodes `cv2.VideoCapture(1)` for the webcam. Without a physical webcam the GUI will launch but the video area will remain blank.

### Key gotchas
- **OpenCV Qt conflict**: Must use `opencv-python-headless` (not `opencv-python`) to avoid Qt plugin conflicts with PyQt5. The non-headless variant bundles its own Qt which crashes with PyQt5.
- **mediapipe version**: Use `mediapipe==0.10.14`. Newer versions (0.10.15+) removed `mp.solutions.selfie_segmentation` which this project depends on.
- **dlib compilation**: Building dlib requires `python3.12-dev`, `cmake`, and `g++`. The default compiler (Clang) may need `libstdc++.so` symlink or GCC must be set as default via `update-alternatives`.
- **Data files**: `shape_predictor_68_face_landmarks.dat` and the `./images/` directory with overlay PNGs are already present in the repo and must not be deleted.

### Testing
- Run `python3 test_core.py` to verify all core functionality (imports, face detection, background effects, fisheye filter, image saving) without a webcam.
- Dependencies are listed in `requirements.txt`.
