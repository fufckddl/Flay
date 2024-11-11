#얼굴 탐지 및 추적
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
tracker = dlib.correlation_tracker()

#프레임에서 얼굴을 탐지하고 랜드마크를 예측하여 얼굴의 중앙 좌표와 폭을 계산
def detect_face(frame):
    faces = detector(frame)
    if faces:
        face = faces[0]
        dlib_shape = predictor(frame, face)
        shape = np.array([[p.x, p.y] for p in dlib_shape.parts()])
        center_X, center_Y = np.mean(shape, axis=0).astype(int)
        face_width = face.width()
        tracker.start_track(frame, face)
        tracking_face = True
    else:
        tracking_face = False
    return face, shape, center_X, center_Y, face_width, tracking_face

#얼굴위치 지속적으로 업데이트
def update_tracker(frame):
    tracker.update(frame)
    pos = tracker.get_position()
    center_X = int((pos.left() + pos.right()) / 2)
    center_Y = int((pos.top() + pos.bottom()) / 2)
    face_width = int(pos.right() - pos.left())
    return center_X, center_Y, face_width
