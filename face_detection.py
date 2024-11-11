import cv2
import numpy as np
import dlib

# 얼굴 탐지기 및 랜드마크 예측기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 얼굴 추적기
tracker = dlib.correlation_tracker()
detector = dlib.get_frontal_face_detector()

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 프레임을 그레이스케일로 변환
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # 얼굴 탐지
    if len(faces) > 0:
        x, y, w, h = faces[0]  # 첫 번째로 감지된 얼굴을 처리
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)  # dlib의 사각형 포맷으로 변환
        dlib_shape = predictor(frame, dlib_rect)  # 랜드마크 예측
        shape = np.array([[p.x, p.y] for p in dlib_shape.parts()])  # 랜드마크 좌표 배열로 변환
        center_X, center_Y = np.mean(shape, axis=0).astype(int)  # 얼굴 중심 계산
        face_width = w
        return dlib_rect, shape, center_X, center_Y, face_width
    else:
        return None, None, None, None, None  # 얼굴을 감지하지 못한 경우

def update_tracker(frame, face):
    # 추적기가 초기화되지 않은 경우
    if tracker.get_position().is_empty():
        # 추적기 시작
        tracker.start_track(frame, face)
        return None, None, None  # 추적을 시작한 경우 위치를 계산하지 않음
    else:
        # 추적기 업데이트
        tracker.update(frame)
        pos = tracker.get_position()  # 추적된 얼굴의 위치
        center_X = int((pos.left() + pos.right()) / 2)
        center_Y = int((pos.top() + pos.bottom()) / 2)
        face_width = int(pos.right() - pos.left())
        return center_X, center_Y, face_width  # 추적된 얼굴의 중심 좌표와 너비 반환
