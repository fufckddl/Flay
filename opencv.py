import cv2
import dlib # 얼굴 검출 라이브러리
import numpy as np
import datetime

# dlib 얼굴 검출기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 고양이 귀와 토끼 귀 이미지 로드 (알파 채널 포함)
cat_ears_image = cv2.imread('image/cat_ears.png', cv2.IMREAD_UNCHANGED)
rabbit_ears_image = cv2.imread('image/rabbit_ears.png', cv2.IMREAD_UNCHANGED)

# 비디오 캡처, 0은 웹 캠 장치를 말한다.
cap = cv2.VideoCapture(1) #0

# 트랙바 콜백 함수, (트랙바 값을 변경할 때마다 호출됨)
def nothing(x):
    pass

# 트랙바 및 창 생성, (영상의 밝기와 대비 조절을 위한 트랙바)
cv2.namedWindow('Face Detection with Filters')
cv2.createTrackbar('Brightness', 'Face Detection with Filters', 50, 100, nothing)
cv2.createTrackbar('Contrast', 'Face Detection with Filters', 50, 100, nothing)

# 필터 적용 상태 및 모드
is_recording = False # 녹화 중인지 확인하는 변수
video_writer = None # 비디오 저장용 객체
apply_filter = False # 필터를 적용할지 여부
mode = 1 # 상태 변수 초기화,  프레임 처리 과정에서 신호를 준다.

# 얼굴에 맞게 고양이 귀, 토끼 귀 위치 조정
def draw_ears(image, ears_image, face):  # 얼굴에 이미지 그리는 함수. image는 프레임, ears_image는 귀 이미지, face 인식된 얼굴 정보
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    ears_width = w
    ears_height = int(ears_width * (ears_image.shape[0] / ears_image.shape[1]))
    ears_x = x
    ears_y = y - ears_height - 5

    if ears_x < 0 or ears_y < 0 or ears_x + ears_width > image.shape[1] or ears_y + ears_height > image.shape[0]:
        return image

    ears_resized = cv2.resize(ears_image, (ears_width, ears_height))
    for c in range(0, 3):
        image[ears_y:ears_y + ears_height, ears_x:ears_x + ears_width, c] = \
            image[ears_y:ears_y + ears_height, ears_x:ears_x + ears_width, c] * \
            (1 - ears_resized[:, :, 3] / 255.0) + \
            ears_resized[:, :, c] * (ears_resized[:, :, 3] / 255.0)
    return image


# 볼록렌즈 필터
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
    ret, frame = cap.read() # 웹캠에서 프레임 읽기
    if not ret:
        break

    frame = cv2.resize(frame, (854, 480)) # 프레임 크기 조정 (저는 노트북 해상도가 낮아서 조절했습니다.)
    frame = cv2.flip(frame, 1 ) #좌우반전

    # 트랙바 값 가져오기
    brightness = cv2.getTrackbarPos('Brightness', 'Face Detection with Filters') - 50
    contrast = cv2.getTrackbarPos('Contrast', 'Face Detection with Filters') / 50.0
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    # 얼굴 검출
    gray_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환(얼굴 검출에 사용)
    faces = detector(gray_frame) # 얼굴 검출

    # 필터 적용
    for face in faces:
        if mode == 2:
            adjusted_frame = draw_ears(adjusted_frame, cat_ears_image, face)
        elif mode == 3:
            adjusted_frame = draw_ears(adjusted_frame, rabbit_ears_image, face)
        # 말풍선
        elif mode == 4:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            bubble_x = x + w + 10
            bubble_y = y - 30
            bubble_width = 200
            bubble_height = 100
            cv2.rectangle(adjusted_frame, (bubble_x, bubble_y), (bubble_x + bubble_width, bubble_y + bubble_height), (255, 255, 255), -1)
            cv2.rectangle(adjusted_frame, (bubble_x, bubble_y), (bubble_x + bubble_width, bubble_y + bubble_height), (0, 0, 0), 2)
            cv2.putText(adjusted_frame, "Hello!", (bubble_x + 10, bubble_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # 볼록렌즈
        elif mode == 5:
            landmarks = predictor(gray_frame, face)
            center, radius = (int((face.left() + face.right()) / 2), int((face.top() + face.bottom()) / 2)), int((face.right() - face.left()) / 2)
            adjusted_frame = apply_fisheye_filter(adjusted_frame, center, radius)

    # 사용법 텍스트 추가
    instructions = (
        "1: Basic Camera  |  2: Cat Ears  |  3: Rabbit Ears  |  4: Speech Bubble  |  5: Fisheye\n"
        "s: Save Image  |  r: Start/Stop Recording"
    )
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
    if key == 27:  # ESC 키로 종료
        break
    elif key == ord('1'): # '1'을 눌러서 기본 카메라 모드
        mode = 1
    elif key == ord('2'): # '2'를 눌러서 고양이 귀 모드
        mode = 2
    elif key == ord('3'): # '3'을 눌러서 토끼 귀 모드
        mode = 3
    elif key == ord('4'): # '4'를 눌러서 말풍선 모드
        mode = 4
    elif key == ord('5'): # '5'를 눌러서 볼록렌즈 모드
        mode = 5
    elif key == ord('s'):  # 이미지 저장
        filename = f"captured_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, adjusted_frame)
        print(f"Image saved at {filename}")

    elif key == ord('r'):  # 녹화 시작/중지
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

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
