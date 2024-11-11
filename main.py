import cv2
from capture_utils import initialize_capture, get_adjusted_frame
from face_detection import detect_face
from filters import apply_filter_mode

# 비디오 캡처 설정 및 창 생성
cap, brightness_trackbar, contrast_trackbar = initialize_capture()
is_recording = False
video_writer = None
mode = 1
frame_count = 0
frame_skip = 3
tracking_face = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (854, 480))
    frame = cv2.flip(frame, 1)
    adjusted_frame = get_adjusted_frame(frame, brightness_trackbar, contrast_trackbar)
    
    # 얼굴 탐지 및 필터 적용
    if frame_count % frame_skip == 0 or tracking_face:
        face, shape, center_X, center_Y, face_width, tracking_face = detect_face(adjusted_frame)
    
    # 모드에 따른 필터 적용
    if tracking_face:
        adjusted_frame = apply_filter_mode(adjusted_frame, mode, face, shape, center_X, center_Y, face_width)
    
    # 화면 표시 및 키 입력 처리
    cv2.imshow('Face Detection with Filters', adjusted_frame)
    key = cv2.waitKey(1) & 0xFF
    # 키 입력 처리 로직 (mode 전환, 녹화 시작/중지 등)
    
    frame_count += 1

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
