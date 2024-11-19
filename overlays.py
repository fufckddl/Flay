import cv2
import numpy as np

# 얼굴 랜드마크를 기준으로 귀 이미지를 얼굴 위에 그리는 함수
def draw_ears(image, ears_image, landmarks):
    # 얼굴 크기 계산을 위한 랜드마크 포인트
    left_eye = landmarks.part(36)  # 왼쪽 눈의 왼쪽 끝점
    right_eye = landmarks.part(45)  # 오른쪽 눈의 오른쪽 끝점
    left_face = landmarks.part(0)   # 얼굴 왼쪽 끝
    right_face = landmarks.part(16) # 얼굴 오른쪽 끝
    
    # 얼굴 전체 너비 계산
    face_width = right_face.x - left_face.x
    
    # 귀 이미지 크기 조정 (얼굴 너비에 비례)
    ears_width = int(face_width * 1.2)  # 얼굴 너비의 1.2배
    ears_height = int(ears_width * (ears_image.shape[0] / ears_image.shape[1]))
    
    # 귀의 위치 계산
    nose_bridge = landmarks.part(27)  # 코 브릿지
    ears_x = nose_bridge.x - (ears_width // 2)
    
    # 얼굴 상단에서 더 위로 올리기 위해 눈썹 위치를 기준으로 설정
    left_eyebrow = landmarks.part(19)  # 왼쪽 눈썹
    right_eyebrow = landmarks.part(24) # 오른쪽 눈썹
    eyebrow_y = min(left_eyebrow.y, right_eyebrow.y)  # 더 높은 눈썹의 y 좌표
    ears_y = eyebrow_y - int(ears_height * 1.0)  # 눈썹에서 위로 올림
    
    try:
        # 귀 이미지 크기 조정
        ears_resized = cv2.resize(ears_image, (ears_width, ears_height))
        
        # 이미지 경계 확인 및 조정
        if ears_x < 0:
            ears_x = 0
        if ears_x + ears_width > image.shape[1]:
            ears_width = image.shape[1] - ears_x
            ears_resized = cv2.resize(ears_image, (ears_width, ears_height))
            
        if ears_y < 0:
            # 상단이 잘리는 경우 이미지의 아래 부분만 표시
            diff = abs(ears_y)
            ears_y = 0
            ears_resized = ears_resized[diff:, :]
            ears_height = ears_resized.shape[0]
            
        if ears_y + ears_height > image.shape[0]:
            ears_height = image.shape[0] - ears_y
            ears_resized = cv2.resize(ears_resized, (ears_width, ears_height))
        
        # 알파 채널이 있는 경우에만 마스크 적용
        if ears_resized.shape[2] == 4:
            # 알파 채널 기반으로 귀 이미지 오버레이
            for c in range(0, 3):  # BGR 채널
                image[ears_y:ears_y + ears_height, ears_x:ears_x + ears_width, c] = \
                    image[ears_y:ears_y + ears_height, ears_x:ears_x + ears_width, c] * \
                    (1 - ears_resized[:, :, 3] / 255.0) + \
                    ears_resized[:, :, c] * (ears_resized[:, :, 3] / 255.0)
    
    except Exception as e:
        pass  # 에러 발생 시 현재 프레임 스킵
    
    return image

# 말풍선 이미지를 얼굴 위에 오버레이하고 텍스트를 삽입하는 함수
def draw_speech_bubble(image, bubble_image, landmarks, text="Hello!"):
    # 얼굴 랜드마크에서 코의 위치를 가져옵니다.
    nose_top = (landmarks.part(30).x, landmarks.part(30).y)
    
    # 말풍선 위치를 코 위쪽에 배치합니다.
    bubble_x = nose_top[0] + 20
    bubble_y = nose_top[1] - 160

    # 말풍선 이미지를 크기에 맞게 조정
    bubble_resized = cv2.resize(bubble_image, (150, 80))
    bubble_height, bubble_width = bubble_resized.shape[:2]

    # 말풍선이 이미지 밖으로 나가지 않도록 제한
    if bubble_x + bubble_width > image.shape[1] or bubble_y + bubble_height > image.shape[0]:
        return image

    # 말풍선 이미지를 얼굴 위에 겹쳐서 오버레이
    for c in range(0, 3):  # BGR 색상 채널에 대해 반복 (알파 채널 제외)
        image[bubble_y:bubble_y + bubble_height, bubble_x:bubble_x + bubble_width, c] = \
            image[bubble_y:bubble_y + bubble_height, bubble_x:bubble_x + bubble_width, c] * \
            (1 - bubble_resized[:, :, 3] / 255.0) + \
            bubble_resized[:, :, c] * (bubble_resized[:, :, 3] / 255.0)  # 알파 채널을 이용해 이미지를 합성

    # 말풍선 내에 텍스트를 추가
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)  # 텍스트 크기 계산
    text_x = bubble_x + (bubble_width - text_size[0]) // 2  # 텍스트 x 좌표 계산
    text_y = bubble_y + (bubble_height + text_size[1]) // 2  # 텍스트 y 좌표 계산
    
    # 텍스트 삽입
    cv2.putText(image, text, (bubble_x + 10, bubble_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return image

# 얼굴에 어안렌즈(피시아이) 효과를 적용하는 함수
def apply_fisheye_filter(frame, center, radius):
    # 원본 이미지 크기에서 x, y 좌표를 그리드 형태로 생성
    map_x, map_y = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))
    
    # 각 픽셀의 중심으로부터의 거리를 계산
    distance = np.sqrt((map_x - center[0]) ** 2 + (map_y - center[1]) ** 2)

    # 왜곡 반경 설정: 반경 내의 영역만 왜곡 효과를 적용
    mask = distance <= radius  # 왜곡이 적용될 영역 마스크 생성
    distortion_factor = distance / radius  # 왜곡 강도 계산
    distortion_factor[mask] = np.minimum(distortion_factor[mask], 1)  # 왜곡 강도를 1로 제한

    # 왜곡된 좌표를 계산
    map_x[mask] = center[0] + (map_x[mask] - center[0]) * distortion_factor[mask]
    map_y[mask] = center[1] + (map_y[mask] - center[1]) * distortion_factor[mask]

    # 왜곡된 이미지를 생성
    distorted_face = cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    # 얼굴 영역만 왜곡을 적용하기 위한 마스크 생성
    face_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
    cv2.circle(face_mask, center, radius, 255, -1)  # 얼굴 영역을 원으로 지정

    face_mask_inv = cv2.bitwise_not(face_mask)  # 얼굴 마스크의 반전 버전

    # 왜곡된 부분과 원본 이미지를 합성
    distorted_face_blurred = cv2.bitwise_and(distorted_face, distorted_face, mask=face_mask)  # 왜곡된 부분
    original_face = cv2.bitwise_and(frame, frame, mask=face_mask_inv)  # 원본 이미지에서 왜곡된 부분 제외

    return cv2.add(distorted_face_blurred, original_face)  # 두 이미지를 합성하여 최종 결과 반환
