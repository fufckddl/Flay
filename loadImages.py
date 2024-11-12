import cv2

# 오버레이 이미지 로드
cat_ears_image = cv2.imread('./images/cat_ears.png', cv2.IMREAD_UNCHANGED)
rabbit_ears_image = cv2.imread('./images/rabbit_ears.png', cv2.IMREAD_UNCHANGED)
speech_bubble_image = cv2.imread('./images/speech_bubble.png', cv2.IMREAD_UNCHANGED)
handsome_overlay = cv2.imread('./images/handsome.png')
bubble_overlay = cv2.imread('./images/work.png')
gym_overlay = cv2.imread('./images/gym.png')