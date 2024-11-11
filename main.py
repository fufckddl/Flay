import numpy as np
import cv2
import dlib

# 얼굴 탐지기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

scaler = 0.5  # 프레임의 크기를 절반으로 축소해 처리 속도 향상
frame_skip = 1  # 매 프레임마다 얼굴을 인식하도록 설정

# 웹캠 사용 설정
cap = cv2.VideoCapture(0)

# 미니언즈 오버레이 이미지 불러오기
overlay = cv2.imread('./images/minion_face_pic.png', cv2.IMREAD_UNCHANGED)

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    if img_to_overlay_t.shape[2] == 4:  # 오버레이에 알파 채널이 있을 때
        b, g, r, a = cv2.split(img_to_overlay_t)
    else:
        return bg_img

    mask = a / 255.0
    mask_inv = 1 - mask

    h, w = img_to_overlay_t.shape[:2]
    top_left_x = int(x - w / 2)
    top_left_y = int(y - h / 2)
    bottom_right_x = int(x + w / 2)
    bottom_right_y = int(y + h / 2)

    top_left_x = max(top_left_x, 0)
    top_left_y = max(top_left_y, 0)
    bottom_right_x = min(bottom_right_x, background_img.shape[1])
    bottom_right_y = min(bottom_right_y, background_img.shape[0])

    roi = bg_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    roi_resized = cv2.resize(img_to_overlay_t, (roi.shape[1], roi.shape[0]))

    b_resized, g_resized, r_resized, a_resized = cv2.split(roi_resized)
    mask_resized = a_resized / 255.0
    mask_inv_resized = 1 - mask_resized

    for c in range(0, 3):
        roi[:, :, c] = (mask_inv_resized * roi[:, :, c] + mask_resized * roi_resized[:, :, c])

    bg_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = roi
    return bg_img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (int(frame.shape[1] * scaler), int(frame.shape[0] * scaler)))
    frame_with_alpha = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2BGRA)
    frame_with_alpha[:, :, 3] = 255

    faces = detector(frame_resized)
    
    if len(faces) > 0:
        face = faces[0]
        dlib_shape = predictor(frame_resized, face)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

        center_X, center_Y = np.mean(shape_2d, axis=0).astype(int)
        top_left = np.min(shape_2d, axis=0)
        bottom_right = np.max(shape_2d, axis=0)
        face_width = bottom_right[0] - top_left[0]
        face_height = bottom_right[1] - top_left[1]

        overlay_width = int(face_width * 2.8)
        overlay_height = int(face_height * 2.5)
        overlay_size = (overlay_width, overlay_height)

        result = overlay_transparent(frame_resized, overlay, center_X, center_Y, overlay_size=overlay_size)
    else:
        # 얼굴이 감지되지 않으면 현재 프레임 그대로 사용
        result = frame_resized

    cv2.imshow('Result with Minion Overlay', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
