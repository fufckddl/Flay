import numpy as np
import cv2
import dlib

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

scaler = 0.2  # 비디오 해상도를 축소하여 처리 속도 향상
frame_skip = 2  # 2프레임마다 처리 (속도 개선)

cap = cv2.VideoCapture('./videos/man.mp4')

# Load the overlay image (minion face)
overlay = cv2.imread('./images/minion_face_pic.png', cv2.IMREAD_UNCHANGED)

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()

    # Resize overlay image if overlay_size is provided
    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    if img_to_overlay_t.shape[2] == 4:  # If overlay has alpha channel
        b, g, r, a = cv2.split(img_to_overlay_t)
    else:
        return bg_img  # If no alpha channel, return the original background

    mask = a / 255.0
    mask_inv = 1 - mask

    # Calculate dimensions of the overlay image
    h, w = img_to_overlay_t.shape[:2]

    # Calculate the region of interest (ROI) in the background image
    top_left_x = int(x - w / 2)
    top_left_y = int(y - h / 2)
    bottom_right_x = int(x + w / 2)
    bottom_right_y = int(y + h / 2)

    # Ensure the ROI is within the bounds of the background image
    top_left_x = max(top_left_x, 0)
    top_left_y = max(top_left_y, 0)
    bottom_right_x = min(bottom_right_x, background_img.shape[1])
    bottom_right_y = min(bottom_right_y, background_img.shape[0])

    # Adjust ROI size to match overlay image
    roi = bg_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    roi_resized = cv2.resize(img_to_overlay_t, (roi.shape[1], roi.shape[0]))

    # Split resized overlay image channels
    b_resized, g_resized, r_resized, a_resized = cv2.split(roi_resized)
    mask_resized = a_resized / 255.0
    mask_inv_resized = 1 - mask_resized

    # Apply overlay to ROI
    for c in range(0, 3):
        roi[:, :, c] = (mask_inv_resized * roi[:, :, c] + mask_resized * roi_resized[:, :, c])

    # Update background image with modified ROI
    bg_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = roi
    return bg_img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 매 n 프레임마다 얼굴 인식 및 랜드마크 추출
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_skip != 0:
        continue  # 처리하지 않고 넘어가기

    # Resize the frame and convert to BGRA (adding alpha channel)
    frame_resized = cv2.resize(frame, (int(frame.shape[1] * scaler), int(frame.shape[0] * scaler)))
    frame_with_alpha = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2BGRA)
    frame_with_alpha[:, :, 3] = 255  # Set alpha channel to 255 (fully opaque)

    # Detect faces
    faces = detector(frame_resized)
    if len(faces) == 0:
        continue  # Skip if no faces are detected

    face = faces[0]
    dlib_shape = predictor(frame_resized, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # Calculate face center
    center_X, center_Y = np.mean(shape_2d, axis=0).astype(int)

    # Calculate face size (adjust overlay size automatically)
    top_left = np.min(shape_2d, axis=0)
    bottom_right = np.max(shape_2d, axis=0)
    face_width = bottom_right[0] - top_left[0]
    face_height = bottom_right[1] - top_left[1]

    # 얼굴 크기를 기준으로 오버레이 이미지 크기 설정
    overlay_width = int(face_width * 2.8)  # 얼굴 너비보다 280% 더 크게 설정
    overlay_height = int(face_height * 2.5)  # 얼굴 높이보다 250% 더 크게 설정
    overlay_size = (overlay_width, overlay_height)

    # 미니언즈 이미지를 얼굴 크기에 맞춰 자동으로 조정하여 오버레이
    result = overlay_transparent(frame_resized, overlay, center_X, center_Y, overlay_size=overlay_size)

    # Show the result
    cv2.imshow('Result with Minion Overlay', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
