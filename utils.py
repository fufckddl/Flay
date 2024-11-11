import cv2
import numpy as np

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

def draw_ears(frame, ear_type, face, shape):
    if ear_type == 'cat':
        ears_image = cv2.imread('./images/cat_ears.png', cv2.IMREAD_UNCHANGED)
    elif ear_type == 'rabbit':
        ears_image = cv2.imread('./images/rabbit_ears.png', cv2.IMREAD_UNCHANGED)
    else:
        return frame  # If unrecognized type, return the original frame

    if ears_image is None:
        print("Error: Unable to load ear images.")
        return frame

    # 얼굴 중앙 계산 (코 대신 얼굴 영역의 중심을 사용)
    center_x = int((face.left() + face.right()) / 2)
    center_y = int((face.top() + face.bottom()) / 2)

    # 귀 위치를 이마보다 위쪽으로 조정 (이마 위치가 아니라 얼굴의 top 위치 기준)
    ear_offset_y = int(face.height() * 0.4)  # 귀를 더 위로 올리기 위한 큰 오프셋 (상단으로 더 많이 이동)
    adjusted_center_y = face.top() + ear_offset_y    # 얼굴의 상단 위치에서 오프셋 적용

    # 귀 이미지를 얼굴 크기에 맞게 조정 (이미지 크기 축소)
    overlay_size = (int(face.width() * 1.5), int(face.height() * 0.6 ))  # 귀 이미지 크기 조정 (얼굴 크기에 비례하여 축소)

    # 조정된 크기와 위치로 귀 이미지 오버레이
    return overlay_transparent(frame, ears_image, center_x, adjusted_center_y, overlay_size=overlay_size)


def apply_fisheye_filter(frame, center, radius):
    map_x, map_y = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))
    distance = np.sqrt((map_x - center[0]) ** 2 + (map_y - center[1]) ** 2)
    
    # 왜곡 반경 설정: 왜곡이 얼굴 영역에만 적용되도록 설정
    mask = distance <= radius
    distortion_factor = distance / radius
    distortion_factor[mask] = np.minimum(distortion_factor[mask], 1)  # 왜곡 강도 제한
    
    # 왜곡된 좌표 계산
    map_x[mask] = center[0] + (map_x[mask] - center[0]) * distortion_factor[mask]
    map_y[mask] = center[1] + (map_y[mask] - center[1]) * distortion_factor[mask]

    # 왜곡된 이미지 생성
    distorted_face = cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    
    # 얼굴 마스크 설정: 왜곡된 얼굴 영역만 처리
    face_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
    cv2.circle(face_mask, center, radius, 255, -1)  # 원형 마스크 적용
    
    face_mask_inv = cv2.bitwise_not(face_mask)
    
    # 왜곡된 부분과 원본 부분을 합성
    distorted_face_blurred = cv2.bitwise_and(distorted_face, distorted_face, mask=face_mask)
    original_face = cv2.bitwise_and(frame, frame, mask=face_mask_inv)
    
    return cv2.add(distorted_face_blurred, original_face)

