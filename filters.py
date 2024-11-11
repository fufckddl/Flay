import cv2
from utils import overlay_transparent, apply_fisheye_filter, draw_ears

def apply_filter_mode(frame, mode, face, shape, center_X, center_Y, face_width):
    # 얼굴 크기 및 중심에 맞게 필터 크기 및 위치 조정
    if mode == 2:  # Cat ears
        frame = draw_ears(frame, 'cat', face, shape)
    elif mode == 3:  # Rabbit ears
        frame = draw_ears(frame, 'rabbit', face, shape)
    elif mode == 4:  # 텍스트 추가
        # 텍스트 위치를 얼굴의 중심 근처로 설정
        cv2.putText(frame, "Hello!", (center_X + 20, center_Y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    elif mode == 5:  # Fisheye 효과
        # 피쉬아이 효과의 반경을 얼굴의 크기에 맞추기
        frame = apply_fisheye_filter(frame, (center_X, center_Y), int(face_width / 2))
    return frame
