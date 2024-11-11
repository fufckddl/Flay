#필터 모드 설정
import cv2
from utils import overlay_transparent, apply_fisheye_filter, draw_ears

def apply_filter_mode(frame, mode, face, shape, center_X, center_Y, face_width):
    if mode == 2:
        frame = draw_ears(frame, 'cat', face, shape)
    elif mode == 3:
        frame = draw_ears(frame, 'rabbit', face, shape)
    elif mode == 4:
        cv2.putText(frame, "Hello!", (center_X + 20, center_Y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    elif mode == 5:
        frame = apply_fisheye_filter(frame, (center_X, center_Y), int(face_width / 2))
    return frame
