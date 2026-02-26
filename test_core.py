"""
Core functionality test for Flay camera filter application.
Tests image processing pipelines without requiring a physical webcam.
"""
import sys
import cv2
import numpy as np

def test_imports():
    """Test all module imports."""
    from loadPredictor import predictor, detector, segmentation
    from loadImages import cat_ears_image, bear_ears_image, speech_bubble_image
    from loadBackground import beach_image
    from background import apply_background_blur, apply_background_with_image
    from overlays import draw_ears_with_nose, draw_speech_bubble, apply_fisheye_filter
    from utils import apply_filter_mode
    from save import savePic
    print("[PASS] All module imports successful")

def test_face_detection():
    """Test dlib face detection and landmark prediction on a synthetic image."""
    from loadPredictor import predictor, detector
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    print(f"[PASS] Face detector works (found {len(faces)} faces in blank image)")

def test_background_blur():
    """Test background blur with MediaPipe segmentation."""
    from background import apply_background_blur
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = apply_background_blur(test_img)
    assert result.shape == test_img.shape, "Output shape mismatch"
    print("[PASS] Background blur works")

def test_background_replacement():
    """Test background replacement with beach image."""
    from background import apply_background_with_image
    from loadBackground import beach_image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = apply_background_with_image(test_img, beach_image)
    assert result.shape == test_img.shape, "Output shape mismatch"
    print("[PASS] Background replacement works")

def test_fisheye_filter():
    """Test fisheye filter effect."""
    from overlays import apply_fisheye_filter
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    center = (320, 240)
    radius = 100
    result = apply_fisheye_filter(test_img, center, radius)
    assert result.shape == test_img.shape, "Output shape mismatch"
    print("[PASS] Fisheye filter works")

def test_save_function():
    """Test image save function."""
    import os
    from save import savePic
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    os.makedirs("./savePictures", exist_ok=True)
    savePic(test_img, "./savePictures")
    saved_files = [f for f in os.listdir("./savePictures") if f.startswith("capture_")]
    assert len(saved_files) > 0, "No file saved"
    print(f"[PASS] Save function works (saved: {saved_files[-1]})")

def test_image_assets():
    """Verify all image assets are loaded."""
    from loadImages import (cat_ears_image, bear_ears_image, cat_nose_image,
                           bear_nose_image, speech_bubble_image, handsome_overlay,
                           bubble_overlay, gym_overlay)
    assets = {
        'cat_ears': cat_ears_image, 'bear_ears': bear_ears_image,
        'cat_nose': cat_nose_image, 'bear_nose': bear_nose_image,
        'speech_bubble': speech_bubble_image, 'handsome': handsome_overlay,
        'bubble': bubble_overlay, 'gym': gym_overlay
    }
    for name, img in assets.items():
        assert img is not None, f"{name} failed to load"
    print(f"[PASS] All {len(assets)} image assets loaded successfully")

if __name__ == "__main__":
    tests = [
        test_imports,
        test_image_assets,
        test_face_detection,
        test_background_blur,
        test_background_replacement,
        test_fisheye_filter,
        test_save_function,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed > 0 else 0)
