#얼굴인식 및 필터, 렌즈 함수

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

def draw_ears(image, ears_image, face, shape):
    # Get the positions of the left and right eyebrows
    left_eyebrow = shape[17:22]
    right_eyebrow = shape[22:27]

    # Calculate the center point above the eyes for positioning the ears
    top_left_x = min(left_eyebrow[:, 0])
    top_right_x = max(right_eyebrow[:, 0])
    center_x = (top_left_x + top_right_x) // 2
    center_y = min(np.min(left_eyebrow[:, 1]), np.min(right_eyebrow[:, 1])) - 20  # Adjusted height above eyebrows

    # Set the ear overlay size based on face width
    face_width = face.width()
    overlay_size = (int(face_width * 1.5), int(face_width * 1.2))  # Adjust overlay size as needed

    # Apply overlay_transparent function for the ears
    image = overlay_transparent(image, ears_image, center_x, center_y, overlay_size=overlay_size)
    return image

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