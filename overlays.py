import cv2
import numpy as np

def draw_ears(image, ears_image, landmarks):
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    mid_eye_x = (left_eye[0] + right_eye[0]) // 2
    mid_eye_y = (left_eye[1] + right_eye[1]) // 2

    ears_width = abs(left_eye[0] - right_eye[0]) * 2
    ears_height = int(ears_width * (ears_image.shape[0] / ears_image.shape[1]))
    ears_x = mid_eye_x - (ears_width // 2)
    ears_y = mid_eye_y - ears_height - 20

    if ears_x < 0 or ears_y < 0 or ears_x + ears_width > image.shape[1] or ears_y + ears_height > image.shape[0]:
        return image

    ears_resized = cv2.resize(ears_image, (ears_width, ears_height))
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
    map_x, map_y = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))
    distance = np.sqrt((map_x - center[0]) ** 2 + (map_y - center[1]) ** 2)
    
    # 왜곡 반경 설정: 왜곡이 얼굴 영역에만 적용되도록 설정
    mask = distance <= radius
    distortion_factor = distance / radius
    distortion_factor[mask] = np.minimum(distortion_factor[mask], 1)  # 왜곡 강도 제한
    
    # 왜곡된 좌표 계산
    map_x[mask] = center[0] + (map_x[mask] - center[0]) * distortion_factor[mask]
    map_y[mask] = center[1] + (map_y[mask] - center[1]) * distortion_factor[mask]

    # 왜곡된 이미지 생성
    distorted_face = cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    
    # 얼굴 마스크 설정: 왜곡된 얼굴 영역만 처리
    face_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
    cv2.circle(face_mask, center, radius, 255, -1)  # 원형 마스크 적용
    
    face_mask_inv = cv2.bitwise_not(face_mask)
    
    # 왜곡된 부분과 원본 부분을 합성
    distorted_face_blurred = cv2.bitwise_and(distorted_face, distorted_face, mask=face_mask)
    original_face = cv2.bitwise_and(frame, frame, mask=face_mask_inv)
    
    return cv2.add(distorted_face_blurred, original_face)


