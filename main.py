import cv2
import dlib
import numpy as np
import datetime

# 얼굴 탐지기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
tracker = dlib.correlation_tracker()

# 고양이 귀, 토끼 귀, 미니언 얼굴 오버레이 이미지 로드 (알파 채널 포함)
cat_ears_image = cv2.imread('./images/cat_ears.png', cv2.IMREAD_UNCHANGED)
rabbit_ears_image = cv2.imread('./images/rabbit_ears.png', cv2.IMREAD_UNCHANGED)
if cat_ears_image is None:
    print("에러: 고양이 귀 이미지가 없습니다.")
if rabbit_ears_image is None:
    print("에러: 토끼 귀 이미지가 없습니다.")
overlay = cv2.imread('./images/minion_face_pic.png', cv2.IMREAD_UNCHANGED)()

if overlay is None:
    print("미니언즈 이미지가 없습니다.")
# 비디오 캡처 설정
cap = cv2.VideoCapture(0)

# 트랙바 및 창 생성 (밝기와 대비 조절용)
cv2.namedWindow('Face Detection with Filters')
cv2.createTrackbar('Brightness', 'Face Detection with Filters', 50, 100, lambda x: x)
cv2.createTrackbar('Contrast', 'Face Detection with Filters', 50, 100, lambda x: x)

# 필터 적용 상태 및 모드
is_recording = False
video_writer = None
apply_filter = False
mode = 1
frame_count = 0
frame_skip = 3  # 얼굴 탐지 빈도
tracking_face = False

# 오버레이 함수 정의
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
    if img_to_overlay_t.shape[2] == 4:
        b, g, r, a = cv2.split(img_to_overlay_t)
        mask = a / 255.0
        mask_inv = 1 - mask
        h, w = img_to_overlay_t.shape[:2]
        top_left_x = max(int(x - w / 2), 0)
        top_left_y = max(int(y - h / 2), 0)
        roi = bg_img[top_left_y:top_left_y + h, top_left_x:top_left_x + w]
        if roi.shape[:2] != (h, w):
            h, w = roi.shape[:2]
            img_to_overlay_t = cv2.resize(img_to_overlay_t, (w, h))
            mask = cv2.resize(mask, (w, h))
            mask_inv = cv2.resize(mask_inv, (w, h))
        for c in range(3):
            roi[:, :, c] = (mask_inv * roi[:, :, c] + mask * img_to_overlay_t[:, :, c])
        bg_img[top_left_y:top_left_y + h, top_left_x:top_left_x + w] = roi
    return bg_img

# 고양이 귀, 토끼 귀 오버레이 함수
def draw_ears(image, ears_image, face):
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    ears_width = w
    ears_height = int(ears_width * (ears_image.shape[0] / ears_image.shape[1]))
    ears_x = x + w // 2  # 얼굴의 중심 x 좌표
    ears_y = y - ears_height // 2  # 얼굴의 중심 y 좌표 위쪽

    # overlay_transparent 함수를 사용하여 오버레이 적용
    image = overlay_transparent(image, ears_image, ears_x, ears_y, overlay_size=(ears_width, ears_height))
    return image

# 피쉬아이 필터 함수
def apply_fisheye_filter(frame, center, radius):
    map_x, map_y = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))
    distance = np.sqrt((map_x - center[0]) ** 2 + (map_y - center[1]) ** 2)
    mask = distance <= radius
    factor = 1
    distortion_factor = factor * (distance / radius)
    map_x[mask] = center[0] + (map_x[mask] - center[0]) * distortion_factor[mask]
    map_y[mask] = center[1] + (map_y[mask] - center[1]) * distortion_factor[mask]
    distorted_face = cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    face_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
    cv2.circle(face_mask, center, radius, 255, -1)
    face_mask_inv = cv2.bitwise_not(face_mask)
    distorted_face_blurred = cv2.bitwise_and(distorted_face, distorted_face, mask=face_mask)
    original_face = cv2.bitwise_and(frame, frame, mask=face_mask_inv)
    return cv2.add(distorted_face_blurred, original_face)

# 메인 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (854, 480))
    frame = cv2.flip(frame, 1)

    # 밝기 및 대비 조절
    brightness = cv2.getTrackbarPos('Brightness', 'Face Detection with Filters') - 50
    contrast = cv2.getTrackbarPos('Contrast', 'Face Detection with Filters') / 50.0
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    # 얼굴 탐지
    if frame_count % frame_skip == 0:
        faces = detector(adjusted_frame)
        if faces:
            face = faces[0]
            dlib_shape = predictor(adjusted_frame, face)
            shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
            center_X, center_Y = np.mean(shape_2d, axis=0).astype(int)
            face_width = face.width()
            face_height = face.height()
            overlay_size = (int(face_width * 2.8), int(face_height * 2.5))
            tracker.start_track(adjusted_frame, face)
            tracking_face = True
        else:
            tracking_face = False
    elif tracking_face:
        tracker.update(adjusted_frame)
        pos = tracker.get_position()
        center_X = int((pos.left() + pos.right()) / 2)
        center_Y = int((pos.top() + pos.bottom()) / 2)
        face_width = int(pos.right() - pos.left())
        overlay_size = (int(face_width * 2.8), int(face_width * 2.5))

    # 모드에 따른 필터 적용
    if tracking_face:
        if mode == 2:
            adjusted_frame = draw_ears(adjusted_frame, cat_ears_image, face)
        elif mode == 3:
            adjusted_frame = draw_ears(adjusted_frame, rabbit_ears_image, face)
        elif mode == 4:
            cv2.rectangle(adjusted_frame, (center_X + 10, center_Y - 30), (center_X + 210, center_Y + 70), (255, 255, 255), -1)
            cv2.putText(adjusted_frame, "Hello!", (center_X + 20, center_Y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        elif mode == 5:
            adjusted_frame = apply_fisheye_filter(adjusted_frame, (center_X, center_Y), int(face_width / 2))

    # 사용법 텍스트 표시
    instructions = "1: Camera  2: Cat Ears  3: Rabbit Ears  4: Bubble  5: Fisheye\ns: Save Image  r: Start/Stop Recording"
    y0, dy = 20, 25
    for i, line in enumerate(instructions.split('\n')):
        y = y0 + i * dy
        cv2.putText(adjusted_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 녹화 중이면 저장
    if is_recording and video_writer:
        video_writer.write(adjusted_frame)

    # 화면 표시
    cv2.imshow('Face Detection with Filters', adjusted_frame)

    # 키 입력 처리
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
    elif key == ord('s'):
        filename = f"captured_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, adjusted_frame)
        print(f"Image saved at {filename}")
    elif key == ord('r'):
        if not is_recording:
            video_filename = f"recorded_video_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = 20.0
            frame_size = (adjusted_frame.shape[1], adjusted_frame.shape[0])
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
            print(f"Recording started: {video_filename}")
            is_recording = True
        else:
            is_recording = False
            video_writer.release()
            print(f"Recording stopped. Video saved at {video_filename}")

    frame_count += 1

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
