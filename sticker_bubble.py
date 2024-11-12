#재혁+나


import cv2
import dlib
import numpy as np

# 얼굴 탐지기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 오버레이 이미지 로드
cat_ears_image = cv2.imread('./images/cat_ears.png', cv2.IMREAD_UNCHANGED)
rabbit_ears_image = cv2.imread('./images/rabbit_ears.png', cv2.IMREAD_UNCHANGED)
speech_bubble_image = cv2.imread('./images/speech_bubble.png', cv2.IMREAD_UNCHANGED)
bubble_overlay = cv2.imread('./images/work.png')
handsome_overlay = cv2.imread('./images/handsome.png')
gym_overlay = cv2.imread('./images/gym.png')

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)
mode = 1
current_overlay = None
use_text_overlay = False

def create_mask(overlay):
    gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_overlay, 240, 255, cv2.THRESH_BINARY_INV)
    return mask

def resize_overlay(face_width, overlay):
    scale_factor = face_width / overlay.shape[1]
    new_size = (int(overlay.shape[1] * scale_factor), int(overlay.shape[0] * scale_factor))
    return cv2.resize(overlay, new_size, interpolation=cv2.INTER_AREA)

def overlay_image(img, overlay, x, y, mask):
    h, w = overlay.shape[:2]
    if x + w > img.shape[1] or y + h > img.shape[0] or x < 0 or y < 0:
        return img
    roi = img[y:y+h, x:x+w]
    img[y:y+h, x:x+w] = np.where(mask[:, :, np.newaxis] == 0, roi, overlay)
    return img

def draw_ears(image, ears_image, landmarks):
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    mid_eye_x = (left_eye[0] + right_eye[0]) // 2
    mid_eye_y = (left_eye[1] + right_eye[1]) // 2

    ears_width = abs(left_eye[0] - right_eye[0]) * 2
    ears_height = int(ears_width * (ears_image.shape[0] / ears_image.shape[1]))
    ears_x = mid_eye_x - (ears_width // 2)
    ears_y = mid_eye_y - ears_height - 20

    if ears_x < 0 or ears_y < 0 or ears_x + ears_width > image.shape[1] or ears_y + ears_height > image.shape[0]:
        return image

    ears_resized = cv2.resize(ears_image, (ears_width, ears_height))
    for c in range(0, 3):
        image[ears_y:ears_y + ears_height, ears_x:ears_x + ears_width, c] = \
            image[ears_y:ears_y + ears_height, ears_x:ears_x + ears_width, c] * \
            (1 - ears_resized[:, :, 3] / 255.0) + \
            ears_resized[:, :, c] * (ears_resized[:, :, 3] / 255.0)
    return image

def draw_speech_bubble(image, bubble_image, landmarks, text="Hello!"):
    nose_top = (landmarks.part(30).x, landmarks.part(30).y)
    bubble_x = nose_top[0] + 20
    bubble_y = nose_top[1] - 160

    bubble_resized = cv2.resize(bubble_image, (150, 80))
    bubble_height, bubble_width = bubble_resized.shape[:2]

    if bubble_x + bubble_width > image.shape[1] or bubble_y + bubble_height > image.shape[0]:
        return image

    for c in range(0, 3):
        image[bubble_y:bubble_y + bubble_height, bubble_x:bubble_x + bubble_width, c] = \
            image[bubble_y:bubble_y + bubble_height, bubble_x:bubble_x + bubble_width, c] * \
            (1 - bubble_resized[:, :, 3] / 255.0) + \
            bubble_resized[:, :, c] * (bubble_resized[:, :, 3] / 255.0)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = bubble_x + (bubble_width - text_size[0]) // 2
    text_y = bubble_y + (bubble_height + text_size[1]) // 2
    cv2.putText(image, text, (bubble_x + 10, bubble_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return image

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
        
        if mode in [2, 3, 4]:  # 1, 2, 3, 4 모드 그룹의 경우
            current_overlay = None
            use_text_overlay = False
            if mode == 2:
                image = draw_ears(image, cat_ears_image, landmarks)
            elif mode == 3:
                image = draw_ears(image, rabbit_ears_image, landmarks)
            elif mode == 4:
                image = draw_speech_bubble(image, speech_bubble_image, landmarks, text="Hello!")
        elif mode in [5, 6, 7, 8]:  # 5, 6, 7, 8 모드 그룹의 경우
            if use_text_overlay:
                balloon_text = "why so serious...."
                (text_width, text_height), baseline = cv2.getTextSize(balloon_text, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 2)
                x_offset = face.left() + (face_width - text_width) // 2
                y_offset = face.top() - text_height - 10
                cv2.putText(image, balloon_text, (x_offset, y_offset), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 255), 2)
            elif current_overlay is not None:
                resized_overlay = resize_overlay(face_width, current_overlay)
                mask = create_mask(resized_overlay)
                overlay_x = face.right() - 10
                overlay_y = face.top() - resized_overlay.shape[0]
                if overlay_x + resized_overlay.shape[1] > image.shape[1]:
                    overlay_x = image.shape[1] - resized_overlay.shape[1]
                if overlay_y < 0:
                    overlay_y = 0
                image = overlay_image(image, resized_overlay, overlay_x, overlay_y, mask)

    instructions = "1:basic/ 2:cat ears/ 3:rabbit ears/ 4:speech bubble/ 5:overlay/ 6:text overlay/ 7:work/ 8:gym"
    cv2.putText(image, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
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
        current_overlay = handsome_overlay
        use_text_overlay = False
    elif key == ord('6'):
        mode = 6
        use_text_overlay = True
        current_overlay = None
    elif key == ord('7'):
        mode = 7
        current_overlay = bubble_overlay
        use_text_overlay = False
    elif key == ord('8'):
        mode = 8
        current_overlay = gym_overlay
        use_text_overlay = False

cap.release()
cv2.destroyAllWindows()
