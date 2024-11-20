import mediapipe as mp
import cv2
import numpy as np

def apply_background_blur(frame):
    try:
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentation:
            results = segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mask = results.segmentation_mask
            mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
            blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
            combined_frame = np.where(mask[..., None] == 255, frame, blurred_frame)
    except Exception as e:
        print(f"MediaPipe 오류: {e}")
        return frame  # 오류 발생 시 원본 프레임 반환
    return combined_frame

def apply_background_with_image(frame, background_image):
    background_image = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentation:
        results = segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = results.segmentation_mask
        mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
        combined_frame = np.where(mask[..., None] == 255, frame, background_image)
    return combined_frame