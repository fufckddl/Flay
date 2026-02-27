import os
import dlib
import cv2
import numpy as np

# 얼굴 탐지기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 배경 분할: Legacy(mp.solutions) 또는 MediaPipe 0.10 Tasks(ImageSegmenter) 중 하나 사용
_segmentation_legacy = None
_segmentation_task = None
_video_timestamp_ms = 0

def _ensure_selfie_model():
    """MediaPipe 0.10용 selfie 세그멘터 모델을 다운로드해 경로 반환."""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "selfie_multiclass_256x256.tflite")
    if os.path.isfile(path):
        return path
    url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
    try:
        import urllib.request
        import ssl
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, context=ctx) as resp, open(path, "wb") as f:
            f.write(resp.read())
        return path
    except Exception as e1:
        try:
            import subprocess
            r = subprocess.run(
                ["curl", "-sL", url, "-o", path],
                capture_output=True,
                timeout=60,
            )
            if r.returncode == 0 and os.path.isfile(path):
                return path
        except Exception:
            pass
        print(f"Selfie segmenter model download failed: {e1}")
        return None

try:
    import mediapipe as mp
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    _segmentation_legacy = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
except AttributeError:
    # MediaPipe 0.10+: Tasks API 사용
    model_path = _ensure_selfie_model()
    if model_path:
        try:
            from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions
            from mediapipe.tasks.python.vision.core import vision_task_running_mode
            from mediapipe.tasks.python.core import base_options
            opts = ImageSegmenterOptions(
                base_options=base_options.BaseOptions(model_asset_path=model_path),
                running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
                output_confidence_masks=True,
                output_category_mask=False,
            )
            _segmentation_task = ImageSegmenter.create_from_options(opts)
        except Exception as e:
            print(f"ImageSegmenter init failed: {e}")
            _segmentation_task = None

def get_person_mask(frame_bgr):
    """
    프레임에서 사람 영역 마스크를 반환합니다. (H, W) uint8, 255=사람.
    사용 가능한 세그멘터가 없으면 None.
    """
    global _video_timestamp_ms
    if _segmentation_legacy is not None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = _segmentation_legacy.process(rgb)
        mask = results.segmentation_mask
        mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
        return mask
    if _segmentation_task is not None:
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            from mediapipe.tasks.python.vision.core import image as mp_image
            mp_img = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)
            result = _segmentation_task.segment_for_video(mp_img, _video_timestamp_ms)
            _video_timestamp_ms += 1
            if result.confidence_masks and len(result.confidence_masks) > 0:
                # mask[0]=배경(background). 사람 = 배경 아님 → (1 - background) > 0.5
                mask_img = result.confidence_masks[0]
                mask_np = np.copy(mask_img.numpy_view()).squeeze()
                h, w = frame_bgr.shape[:2]
                if mask_np.shape[0] != h or mask_np.shape[1] != w:
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
                person = (1.0 - np.clip(mask_np, 0, 1)) > 0.5
                mask = person.astype(np.uint8) * 255
                return mask
        except Exception as e:
            pass
    return None


#haarcascade (사용 x)
#face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
