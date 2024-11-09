import numpy as np
import cv2
import dlib

# 얼굴 탐지기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

scaler = 0.2  # 성능 향상을 위한 영상 크기 축소 비율
frame_skip = 2  # 프레임을 건너뛰는 설정 (2프레임마다 처리)

cap = cv2.VideoCapture('./videos/man.mp4')  # 영상 파일 열기

def apply_fisheye_filter(frame, face):
    # 얼굴의 바운딩 박스 가져오기
    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

    # 얼굴 크기 기반으로 볼록렌즈 효과 파라미터 정의
    center = (x + w // 2, y + h // 2)  # 얼굴 중앙 좌표
    radius = int(min(w, h) * 0.6)  # 얼굴의 반지름을 0.6배 크게 설정

    # 전체 영상에 대해 그리드 맵 생성
    map_x, map_y = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))

    # 얼굴 중앙에서의 거리 계산
    distance = np.sqrt((map_x - center[0])**2 + (map_y - center[1])**2)

    # 얼굴 영역 내에서만 볼록렌즈 효과 적용 (반지름 내 거리)
    mask = distance <= radius  # 얼굴 영역만 마스크로 설정

    # 볼록렌즈 효과에서는 가장자리가 확대되고, 중앙은 축소됨
    factor = 1  # 볼록렌즈 효과의 강도 (1이 가장 적당)
    
    # 얼굴의 중앙은 축소되고, 가장자리는 확대됨 (왜곡 계수 계산)
    distortion_factor = factor * (distance / radius)  # 중앙에서 멀어질수록 왜곡 강도 증가
    map_x[mask] = center[0] + (map_x[mask] - center[0]) * distortion_factor[mask]
    map_y[mask] = center[1] + (map_y[mask] - center[1]) * distortion_factor[mask]

    # 왜곡된 얼굴 영역만 추출
    distorted_face = cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    # 얼굴 영역의 마스크 생성 (원형 마스크)
    face_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)  # 원형 마스크 생성
    cv2.circle(face_mask, center, radius, 255, -1)  # 얼굴 영역만 흰색 원으로 설정

    # 얼굴 영역 외의 마스크 생성 (반대 마스크)
    face_mask_inv = cv2.bitwise_not(face_mask)

    # 얼굴 영역에 볼록렌즈 효과 적용
    distorted_face_blurred = cv2.bitwise_and(distorted_face, distorted_face, mask=face_mask)  # 얼굴 영역에만 볼록렌즈 적용
    original_face = cv2.bitwise_and(frame, frame, mask=face_mask_inv)  # 얼굴 영역 외의 부분은 원본 영상 유지

    # 왜곡된 얼굴과 원본 얼굴을 합침
    frame_blended = cv2.add(distorted_face_blurred, original_face)

    return frame_blended

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 매 n 프레임마다 얼굴 인식 및 랜드마크 추출
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_skip != 0:
        continue

    # 성능 향상을 위해 영상 크기 축소
    frame_resized = cv2.resize(frame, (int(frame.shape[1] * scaler), int(frame.shape[0] * scaler)))

    # 얼굴 인식
    faces = detector(frame_resized)
    if len(faces) == 0:
        continue

    # 얼굴마다 볼록렌즈 효과 적용
    for face in faces:
        frame_resized = apply_fisheye_filter(frame_resized, face)

    # 결과 출력
    cv2.imshow('Concave Filter on Face (Reverse Fisheye)', frame_resized)

    # 'q' 키를 눌러서 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 비디오 캡처 해제
cv2.destroyAllWindows()  # 모든 OpenCV 윈도우 종료
