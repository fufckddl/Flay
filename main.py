import cv2
from capture_utils import initialize_capture, get_adjusted_frame
from face_detection import detect_face, update_tracker  # update_tracker를 추가로 임포트
from filters import apply_filter_mode
import dlib  # dlib import 추가

# Create the display window first
cv2.namedWindow('Face Detection with Filters')

# Now initialize capture and other elements
cap, brightness_trackbar, contrast_trackbar = initialize_capture()
is_recording = False
video_writer = None
mode = 1
frame_count = 0
frame_skip = 5  # 얼굴 탐지 간격을 늘려 성능 개선
tracking_face = False

# 추적 상태 유지
tracker_initialized = False
tracker = dlib.correlation_tracker()  # tracker 객체 생성

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if cv2.getWindowProperty('Face Detection with Filters', cv2.WND_PROP_VISIBLE) < 1:
        break
    
    frame = cv2.resize(frame, (640, 360))  # 프레임 크기를 더 작게 조정
    frame = cv2.flip(frame, 1)
    adjusted_frame = get_adjusted_frame(frame, brightness_trackbar, contrast_trackbar)
    
    # 얼굴 탐지 및 필터 적용
    if frame_count % frame_skip == 0 or not tracking_face:  # 얼굴 탐지 간격을 설정
        face, shape, center_X, center_Y, face_width = detect_face(adjusted_frame)
        
        if face is not None:
            tracking_face = True  # 얼굴 탐지되면 추적 상태로 변경
            tracker_initialized = False  # 새로운 얼굴을 탐지하면 추적기 초기화
        else:
            tracking_face = False  # 얼굴이 없으면 추적 상태 False로 유지
    
    # 얼굴 추적
    if tracking_face and not tracker_initialized:
        # 추적기 시작
        tracker.start_track(adjusted_frame, face)
        tracker_initialized = True
    
    # 얼굴 추적기 사용
    if tracking_face and tracker_initialized:
        # update_tracker 호출 시 face도 함께 전달
        try:
            center_X, center_Y, face_width = update_tracker(adjusted_frame, face)
        except Exception as e:
            print(f"Error in update_tracker: {e}")
            tracking_face = False  # 추적 오류 발생 시 추적 상태를 False로 설정
            tracker_initialized = False  # 추적기 초기화
    
    # 모드에 따른 필터 적용
    if tracking_face:
        adjusted_frame = apply_filter_mode(adjusted_frame, mode, face, shape, center_X, center_Y, face_width)
    
    # 기본 카메라 모드일 때 얼굴 경계와 랜드마크 표시
    if mode == 1 and tracking_face:
        # 얼굴 경계 상자 그리기
        cv2.rectangle(adjusted_frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        
        # 랜드마크 포인트 그리기
        for (x, y) in shape:
            cv2.circle(adjusted_frame, (x, y), 2, (0, 0, 255), -1)  # 빨간색으로 랜드마크 점 표시
    
    # 화면 표시 및 키 입력 처리
    cv2.imshow('Face Detection with Filters', adjusted_frame)
    key = cv2.waitKey(1) & 0xFF  # 대기시간을 1로 설정
    
    # 키 입력 처리 로직
    if key == ord('1'):
        mode = 1  # 기본 카메라 모드
    elif key == ord('2'):
        mode = 2  # 고양이 귀 필터
    elif key == ord('3'):
        mode = 3  # 토끼 귀 필터
    elif key == ord('4'):
        mode = 4  # 텍스트 버블
    elif key == ord('5'):
        mode = 5  # 피쉬아이 필터
    elif key == 27:  # ESC 키
        break
    
    frame_count += 1

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
