import numpy as np
import cv2
import dlib

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

scaler = 0.2
frame_skip = 2

cap = cv2.VideoCapture('./videos/man.mp4')

def apply_fisheye_filter(frame, face):
    # Get face bounding box
    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

    # Define fisheye effect parameters based on face size
    center = (x + w // 2, y + h // 2)
    radius = min(w, h) // 2

    # Create mesh grid for the whole frame
    map_x, map_y = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))

    # Calculate distance from center (the face center)
    distance = np.sqrt((map_x - center[0])**2 + (map_y - center[1])**2)

    # Apply fisheye effect only inside the face area (distance inside the radius)
    mask = distance <= radius  # Apply fisheye only inside the face area
    factor = 0.4  # Control the intensity of fisheye effect

    # Calculate distortion effect (we apply the distortion only inside the face area)
    map_x[mask] = map_x[mask] + (map_x[mask] - center[0]) * factor * (1 - distance[mask] / radius)
    map_y[mask] = map_y[mask] + (map_y[mask] - center[1]) * factor * (1 - distance[mask] / radius)

    # Create the distorted face only in the face area
    distorted_face = cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    # Create a mask for the face area (circular mask)
    face_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
    cv2.circle(face_mask, center, radius, 255, -1)  # Circular mask with the face center as the circle's center

    # Create an inverse mask for the outside area (region outside the face)
    face_mask_inv = cv2.bitwise_not(face_mask)

    # Apply the fisheye effect only inside the face area
    distorted_face_blurred = cv2.bitwise_and(distorted_face, distorted_face, mask=face_mask)  # Apply fisheye effect only in the face area
    original_face = cv2.bitwise_and(frame, frame, mask=face_mask_inv)  # Keep original outside the face area

    # Combine the distorted face and the original face
    frame_blended = cv2.add(distorted_face_blurred, original_face)

    return frame_blended

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 매 n 프레임마다 얼굴 인식 및 랜드마크 추출
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_skip != 0:
        continue

    # Resize the frame for performance
    frame_resized = cv2.resize(frame, (int(frame.shape[1] * scaler), int(frame.shape[0] * scaler)))

    # Detect faces
    faces = detector(frame_resized)
    if len(faces) == 0:
        continue

    # Apply fisheye effect to each detected face
    for face in faces:
        frame_resized = apply_fisheye_filter(frame_resized, face)

    # Show the result
    cv2.imshow('Fisheye Filter on Face', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
