import cv2
import numpy as np

def draw_ears(image, ears_image, landmarks):
    # 얼굴 랜드마크에서 눈 위치 가져오기
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    mid_eye_x = (left_eye[0] + right_eye[0]) // 2
    mid_eye_y = (left_eye[1] + right_eye[1]) // 2

    # 귀 이미지 크기 비율 계산
    ears_width = abs(left_eye[0] - right_eye[0]) * 2
    ears_height = int(ears_width * (ears_image.shape[0] / ears_image.shape[1]))
    ears_x = mid_eye_x - (ears_width // 2)
    ears_y = mid_eye_y - ears_height - 20

    # 귀 이미지가 화면 범위를 벗어나지 않도록 제한
    ears_x = max(0, min(ears_x, image.shape[1] - ears_width))
    ears_y = max(0, min(ears_y, image.shape[0] - ears_height))

    # 귀 이미지 크기 조정
    ears_resized = cv2.resize(ears_image, (ears_width, ears_height))

    # 귀 이미지를 얼굴 위에 오버레이
    for c in range(0, 3):
        image[ears_y:ears_y + ears_height, ears_x:ears_x + ears_width, c] = \
            image[ears_y:ears_y + ears_height, ears_x:ears_x + ears_width, c] * \
            (1 - ears_resized[:, :, 3] / 255.0) + \
            ears_resized[:, :, c] * (ears_resized[:, :, 3] / 255.0)
    
    return image

def draw_speech_bubble(image, bubble_image, landmarks, text="Hello!"):
    nose_top = (landmarks.part(30).x, landmarks.part(30).y)
    bubble_x = nose_top[0] + 20
    bubble_y = nose_top[1] - 160

    bubble_resized = cv2.resize(bubble_image, (150, 80))
    bubble_height, bubble_width = bubble_resized.shape[:2]

    if bubble_x + bubble_width > image.shape[1] or bubble_y + bubble_height > image.shape[0]:
        return image

    for c in range(0, 3):
        image[bubble_y:bubble_y + bubble_height, bubble_x:bubble_x + bubble_width, c] = \
            image[bubble_y:bubble_y + bubble_height, bubble_x:bubble_x + bubble_width, c] * \
            (1 - bubble_resized[:, :, 3] / 255.0) + \
            bubble_resized[:, :, c] * (bubble_resized[:, :, 3] / 255.0)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = bubble_x + (bubble_width - text_size[0]) // 2
    text_y = bubble_y + (bubble_height + text_size[1]) // 2
    cv2.putText(image, text, (bubble_x + 10, bubble_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return image

def apply_fisheye_filter(frame, center, radius):
    # 맵 생성 시 필터가 적용될 반경 안의 좌표만 계산
    map_x, map_y = np.meshgrid(
        np.arange(center[0] - radius, center[0] + radius),
        np.arange(center[1] - radius, center[1] + radius)
    )
    
    # 거리 계산 및 왜곡 강도 조정
    distance = np.sqrt((map_x - center[0]) ** 2 + (map_y - center[1]) ** 2)
    mask = distance <= radius
    distortion_factor = np.minimum(distance / radius, 1)
    
    # 필요한 부분만 왜곡하여 적용
    map_x = (center[0] + (map_x - center[0]) * distortion_factor).astype(np.float32)
    map_y = (center[1] + (map_y - center[1]) * distortion_factor).astype(np.float32)
    
    distorted_face = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    # 원본과 왜곡된 부분 합성
    frame[center[1] - radius:center[1] + radius, center[0] - radius:center[0] + radius] = distorted_face
    return frame


