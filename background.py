from loadPredictor import get_person_mask
import cv2
import numpy as np

def apply_background_blur(frame):
    """
    사람 영역을 제외한 나머지 배경에 블러 효과를 적용하여 반환합니다.
    세그멘터를 사용할 수 없으면 원본 프레임을 그대로 반환합니다.
    """
    mask = get_person_mask(frame)
    if mask is None:
        return frame
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
    combined_frame = np.where(mask[..., None] == 255, frame, blurred_frame)
    return combined_frame

def apply_background_with_image(frame, background_image):
    """
    사람 영역을 제외한 나머지 배경에 이미지를 덮어씌워 반환합니다.
    세그멘터를 사용할 수 없으면 원본 프레임을 그대로 반환합니다.
    """
    mask = get_person_mask(frame)
    if mask is None:
        return frame
    background_image = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    combined_frame = np.where(mask[..., None] == 255, frame, background_image)
    return combined_frame
