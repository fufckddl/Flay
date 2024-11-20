import dlib
import mediapipe as mp

# 얼굴 탐지기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# MediaPipe Selfie Segmentation 초기화
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)


#haarcascade (사용 x)
#face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
