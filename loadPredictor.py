import dlib
import cv2

# 얼굴 탐지기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# MediaPipe Selfie Segmentation 초기화 (0.10+ 에서는 solutions 제거됨 → 선택적 사용)
segmentation = None
try:
    import mediapipe as mp
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
except AttributeError:
    pass  # MediaPipe 0.10+ 사용 시 Blur/Beach 배경은 비활성화됨


#haarcascade (사용 x)
#face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
