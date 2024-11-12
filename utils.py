import cv2
import numpy as np
from overlays import draw_ears, draw_speech_bubble, apply_fisheye_filter
from loadImages import cat_ears_image, rabbit_ears_image, speech_bubble_image, handsome_overlay, bubble_overlay, gym_overlay

def create_mask(overlay):
    gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_overlay, 240, 255, cv2.THRESH_BINARY_INV)
    return mask

def resize_overlay(face_width, overlay):
    scale_factor = face_width / overlay.shape[1]
    new_size = (int(overlay.shape[1] * scale_factor), int(overlay.shape[0] * scale_factor))
    return cv2.resize(overlay, new_size, interpolation=cv2.INTER_AREA)

def overlay_image(img, overlay, x, y, mask):
    h, w = overlay.shape[:2]
    if x + w > img.shape[1] or y + h > img.shape[0] or x < 0 or y < 0:
        return img
    roi = img[y:y+h, x:x+w]
    img[y:y+h, x:x+w] = np.where(mask[:, :, np.newaxis] == 0, roi, overlay)
    return img

def apply_filter_mode(image, mode, current_overlay, use_text_overlay, face, landmarks, face_width):
    if mode in [2, 3, 4, 5]:
        current_overlay = None
        use_text_overlay = False
        if mode == 2:
            image = draw_ears(image, cat_ears_image, landmarks)
        elif mode == 3:
            image = draw_ears(image, rabbit_ears_image, landmarks)
        elif mode == 4:
            image = draw_speech_bubble(image, speech_bubble_image, landmarks, text="Hello!")
        elif mode == 5:
            # 얼굴의 중심과 반경을 계산하여 fisheye 필터 적용
            center = (face.left() + face_width // 2, face.top() + face_width // 2)  # 얼굴 중심 계산
            radius = face_width // 2  # 얼굴 너비의 절반을 반경으로 설정
            image = apply_fisheye_filter(image, center, radius)
    elif mode in [6, 7, 8, 9]:
        if use_text_overlay:
            balloon_text = "why so serious...."
            (text_width, text_height), baseline = cv2.getTextSize(balloon_text, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 2)
            x_offset = face.left() + (face_width - text_width) // 2
            y_offset = face.top() - text_height - 10
            cv2.putText(image, balloon_text, (x_offset, y_offset), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 255), 2)
        elif current_overlay is not None:
            resized_overlay = resize_overlay(face_width, current_overlay)
            mask = create_mask(resized_overlay)
            overlay_x = face.right() - 10
            overlay_y = face.top() - resized_overlay.shape[0]
            overlay_x = min(overlay_x, image.shape[1] - resized_overlay.shape[1])
            overlay_y = max(overlay_y, 0)
            image = overlay_image(image, resized_overlay, overlay_x, overlay_y, mask)
    return image