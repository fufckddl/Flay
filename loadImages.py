import cv2
import os

# 이미지 경로 설정
image_path = './images/'

# 이미지 파일들 로드
cat_ears_image = cv2.imread(os.path.join(image_path, 'cat_ears.png'), cv2.IMREAD_UNCHANGED)
rabbit_ears_image = cv2.imread(os.path.join(image_path, 'rabbit_ears.png'), cv2.IMREAD_UNCHANGED)
speech_bubble_image = cv2.imread(os.path.join(image_path, 'speech_bubble.png'), cv2.IMREAD_UNCHANGED)
handsome_overlay = cv2.imread(os.path.join(image_path, 'handsome.png'))
bubble_overlay = cv2.imread(os.path.join(image_path, 'work.png'))
gym_overlay = cv2.imread(os.path.join(image_path, 'gym.png'))

# 이미지 로드 확인
if cat_ears_image is None:
    print("Error loading cat ears image.")
else:
    print("Cat ears image loaded with shape:", cat_ears_image.shape)

if rabbit_ears_image is None:
    print("Error loading rabbit ears image.")
else:
    print("Rabbit ears image loaded with shape:", rabbit_ears_image.shape)

if speech_bubble_image is None:
    print("Error loading speech bubble image.")
else:
    print("Speech bubble image loaded with shape:", speech_bubble_image.shape)

if handsome_overlay is None:
    print("Error loading handsome overlay image.")
else:
    print("Handsome overlay image loaded with shape:", handsome_overlay.shape)

if bubble_overlay is None:
    print("Error loading bubble overlay image.")
else:
    print("Bubble overlay image loaded with shape:", bubble_overlay.shape)

if gym_overlay is None:
    print("Error loading gym overlay image.")
else:
    print("Gym overlay image loaded with shape:", gym_overlay.shape)
