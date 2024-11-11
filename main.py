import numpy as np
import cv2
import dlib

# 얼굴 탐지기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
tracker = dlib.correlation_tracker()  # 얼굴 추적기 초기화

# 프레임 크기 및 프레임 스킵 설정
scaler = 0.5  # 얼굴 인식할 때 프레임을 축소하여 성능 향상
frame_skip = 3  # 3프레임에 한 번씩 얼굴 탐지 수행

# 웹캠 설정
cap = cv2.VideoCapture(0)
overlay = cv2.imread('./images/minion_face_pic.png', cv2.IMREAD_UNCHANGED)

# 오버레이 함수 정의
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()

    # 오버레이 이미지 크기 조정
    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # 알파 채널이 있는 경우에만 오버레이 수행
    if img_to_overlay_t.shape[2] == 4:
        b, g, r, a = cv2.split(img_to_overlay_t)
        mask = a / 255.0
        mask_inv = 1 - mask

        # 오버레이 위치와 크기 계산
        h, w = img_to_overlay_t.shape[:2]
        top_left_x = max(int(x - w / 2), 0)
        top_left_y = max(int(y - h / 2), 0)

        # `roi` 크기와 `img_to_overlay_t`의 크기를 맞춰 조정
        roi = bg_img[top_left_y:top_left_y + h, top_left_x:top_left_x + w]

        if roi.shape[:2] != (h, w):
            # roi의 크기를 img_to_overlay_t와 동일하게 맞추기
            h, w = roi.shape[:2]
            img_to_overlay_t = cv2.resize(img_to_overlay_t, (w, h))
            mask = cv2.resize(mask, (w, h))
            mask_inv = cv2.resize(mask_inv, (w, h))

        # 오버레이 적용
        for c in range(3):
            roi[:, :, c] = (mask_inv * roi[:, :, c] + mask * img_to_overlay_t[:, :, c])

        bg_img[top_left_y:top_left_y + h, top_left_x:top_left_x + w] = roi
    return bg_img


frame_count = 0  # 프레임 카운트 초기화
tracking_face = False  # 얼굴 추적 여부
result = None  # result 변수 초기화

# 메인 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 얼굴 인식을 위한 프레임 크기 축소
    frame_resized = cv2.resize(frame, (int(frame.shape[1] * scaler), int(frame.shape[0] * scaler)))

    # 얼굴 탐지를 프레임 스킵 간격으로 수행하여 부드러움 향상
    if frame_count % frame_skip == 0:
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

            # 얼굴 추적 시작
            tracker.start_track(frame_resized, face)
            tracking_face = True
        else:
            result = frame_resized  # 얼굴이 없으면 원본 프레임 유지
    elif tracking_face:
        # 얼굴 추적
        tracker.update(frame_resized)
        pos = tracker.get_position()

        # 추적된 얼굴의 중앙 좌표 계산
        center_X = int((pos.left() + pos.right()) / 2)
        center_Y = int((pos.top() + pos.bottom()) / 2)

        # 얼굴 크기 계산
        face_width = int(pos.right() - pos.left())
        face_height = int(pos.bottom() - pos.top())

        overlay_width = int(face_width * 2.8)
        overlay_height = int(face_height * 2.5)
        overlay_size = (overlay_width, overlay_height)

        result = overlay_transparent(frame_resized, overlay, center_X, center_Y, overlay_size=overlay_size)
    else:
        result = frame_resized  # 얼굴 추적 중이지 않으면 원본 프레임 사용

    # 원본 크기로 확장하여 표시
    if result is not None:  # result가 None일 때 오류 방지
        result = cv2.resize(result, (frame.shape[1], frame.shape[0]))
        cv2.imshow('Result with Minion Overlay', result)

    frame_count += 1  # 프레임 카운트 증가

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
