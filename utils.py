import cv2
import numpy as np
from overlays import draw_ears, draw_speech_bubble, apply_fisheye_filter
from loadImages import cat_ears_image, rabbit_ears_image, speech_bubble_image, handsome_overlay, bubble_overlay, gym_overlay

# 마스크 생성 함수: 오버레이 이미지의 흑백 마스크를 만들어서 투명한 부분을 식별
def create_mask(overlay):
    gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)  # 오버레이를 그레이스케일로 변환
    _, mask = cv2.threshold(gray_overlay, 240, 255, cv2.THRESH_BINARY_INV)  # 밝은 부분을 투명하게 처리
    return mask

# 얼굴 크기에 맞춰 오버레이 이미지의 크기를 조정하는 함수
def resize_overlay(face_width, overlay):
    scale_factor = face_width / overlay.shape[1]  # 얼굴 크기 대비 오버레이 크기 비율 계산
    new_size = (int(overlay.shape[1] * scale_factor), int(overlay.shape[0] * scale_factor))  # 새 크기 계산
    return cv2.resize(overlay, new_size, interpolation=cv2.INTER_AREA)  # 이미지 크기 변경

# 오버레이 이미지를 원본 이미지에 합성하는 함수
def overlay_image(img, overlay, x, y, mask):
    h, w = overlay.shape[:2]  # 오버레이 이미지의 높이와 너비
    # 오버레이가 이미지 범위를 벗어나지 않도록 검사
    if x + w > img.shape[1] or y + h > img.shape[0] or x < 0 or y < 0:
        return img
    roi = img[y:y+h, x:x+w]  # 합성할 영역(Region of Interest)
    # 마스크를 이용해 오버레이 이미지 합성
    img[y:y+h, x:x+w] = np.where(mask[:, :, np.newaxis] == 0, roi, overlay)
    return img

# 다양한 필터 모드를 적용하는 함수
def apply_filter_mode(image, mode, current_overlay, use_text_overlay, face, landmarks, face_width):
    if mode in [2, 3, 4, 5]:
        current_overlay = None
        use_text_overlay = False
        
        if mode == 2:
            # 고양이 귀 추가
            image = draw_ears(image, cat_ears_image, landmarks)  # face 인자 제거
        elif mode == 3:
            # 토끼 귀 추가
            image = draw_ears(image, rabbit_ears_image, landmarks)  # face 인자 제거
        elif mode == 4:
            # 말풍선 추가
            image = draw_speech_bubble(image, speech_bubble_image, landmarks, text="Hello!")
        elif mode == 5:
            # 얼굴 중심에 fisheye 필터 적용
            center = (face.left() + face_width // 2, face.top() + face_width // 2)
            radius = face_width // 2
            image = apply_fisheye_filter(image, center, radius)
    elif mode in [6, 7, 8, 9]:
        if use_text_overlay:
            # 텍스트가 있는 말풍선 오버레이 적용
            balloon_text = "why so serious...."
            (text_width, text_height), baseline = cv2.getTextSize(balloon_text, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 2)  # 텍스트 크기 계산
            x_offset = face.left() + (face_width - text_width) // 2  # 텍스트가 얼굴 중심에 오도록 x 좌표 설정
            y_offset = face.top() - text_height - 10  # 텍스트가 얼굴 위에 위치하도록 y 좌표 설정
            # 텍스트 출력
            cv2.putText(image, balloon_text, (x_offset, y_offset), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 255), 2)
        elif current_overlay is not None:
            # 기존 오버레이가 있으면, 이를 얼굴 크기에 맞춰 적용
            resized_overlay = resize_overlay(face_width, current_overlay)  # 오버레이 크기 변경
            mask = create_mask(resized_overlay)  # 오버레이 마스크 생성
            overlay_x = face.right() - 50  # 오버레이의 x 좌표 계산
            overlay_y = face.top() - resized_overlay.shape[0]  # 오버레이의 y 좌표 계산
            overlay_x = min(overlay_x, image.shape[1] - resized_overlay.shape[1])  # 이미지 범위 벗어나지 않도록 조정
            overlay_y = max(overlay_y, 0)  # 이미지 상단에 오버레이가 위치하도록 조정
            image = overlay_image(image, resized_overlay, overlay_x, overlay_y, mask)  # 오버레이 합성
    return image  # 최종적으로 변경된 이미지를 반환
