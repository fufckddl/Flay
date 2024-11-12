import cv2
import dlib
from utils import apply_filter_mode
from loadImages import cat_ears_image, rabbit_ears_image, speech_bubble_image, handsome_overlay, bubble_overlay, gym_overlay


# 얼굴 탐지기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)
mode = 1
current_overlay = None
use_text_overlay = False

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.flip(image, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        face_width = face.right() - face.left()
        
        image = apply_filter_mode(image, mode, current_overlay, use_text_overlay, face, landmarks, face_width)
    '''
    TODO
    - 화면에 텍스트를 띄우는 방식이 아닌 UI로 진행
    '''
    #instructions = "1:basic/ 2:cat ears/ 3:rabbit ears/ 4:speech bubble/ 5:overlay/ 6:text overlay/ 7:work/ 8:gym"
    #cv2.putText(image, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Face Detection with Effects", image)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('1'):
        mode = 1
    elif key == ord('2'):
        mode = 2
    elif key == ord('3'):
        mode = 3
    elif key == ord('4'):
        mode = 4
    elif key == ord('5'):
        mode = 5
    elif key == ord('6'):
        mode = 6
        current_overlay = handsome_overlay
        use_text_overlay = False
    elif key == ord('7'):
        mode = 7
        use_text_overlay = True
        current_overlay = None
    elif key == ord('8'):
        mode = 7
        current_overlay = bubble_overlay
        use_text_overlay = False
    elif key == ord('9'):
        mode = 9
        current_overlay = gym_overlay
        use_text_overlay = False

cap.release()
cv2.destroyAllWindows()
