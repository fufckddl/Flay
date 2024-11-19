import cv2
import dlib
import numpy as np

# 얼굴 탐지기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 

# 오버레이 이미지 로드
cat_ears_image = cv2.imread('cat_ears.png', cv2.IMREAD_UNCHANGED)
rabbit_ears_image = cv2.imread('rabbit_ears.png', cv2.IMREAD_UNCHANGED)
speech_bubble_image = cv2.imread('speech_bubble.png', cv2.IMREAD_UNCHANGED)
bubble_overlay = cv2.imread('work.png')
handsome_overlay = cv2.imread('handsome.png')
gym_overlay = cv2.imread('muscle.jpg')

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0) 
mode = 1  # 초기 모드 설정
current_overlay = None  # 현재 적용할 오버레이
use_text_overlay = False  # 텍스트 오버레이 사용 여부

# 오버레이 이미지에서 마스크 생성 (알파 채널 사용)
def create_mask(overlay):
    gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)  # 오버레이 이미지를 그레이스케일로 변환
    _, mask = cv2.threshold(gray_overlay, 240, 255, cv2.THRESH_BINARY_INV)  # 배경을 검정색, 전경을 흰색으로 마스크 생성
    return mask

# 얼굴 너비에 맞게 오버레이 이미지 크기 조정
def resize_overlay(face_width, overlay):
    scale_factor = face_width / overlay.shape[1]  # 얼굴 너비에 맞춰 스케일 비율 계산
    new_size = (int(overlay.shape[1] * scale_factor), int(overlay.shape[0] * scale_factor))  # 새로운 크기 계산
    return cv2.resize(overlay, new_size, interpolation=cv2.INTER_AREA)  # 오버레이 크기 변경

# 오버레이 이미지를 원본 이미지 위에 겹쳐서 표시
def overlay_image(img, overlay, x, y, mask):
    h, w = overlay.shape[:2]  # 오버레이 이미지의 높이와 너비
    if x + w > img.shape[1]:
        x = img.shape[1] - w  # 이미지 오른쪽 끝을 벗어나지 않도록 조정
    if y + h > img.shape[0]:
        y = img.shape[0] - h  # 이미지 아래쪽 끝을 벗어나지 않도록 조정
    if x < 0:
        x = 0  # 이미지 왼쪽 끝을 벗어나지 않도록 조정
    if y < 0:
        y = 0  # 이미지 위쪽 끝을 벗어나지 않도록 조정

    roi = img[y:y+h, x:x+w]  # 겹칠 영역
    img[y:y+h, x:x+w] = np.where(mask[:, :, np.newaxis] == 0, roi, overlay)  # 마스크를 이용하여 오버레이 적용
    return img

# 고양이 귀 오버레이 그리기
def draw_ears(image, ears_image, landmarks):
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)  # 왼쪽 눈 좌표
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)  # 오른쪽 눈 좌표
    mid_eye_x = (left_eye[0] + right_eye[0]) // 2  # 두 눈의 중간 x 좌표
    mid_eye_y = (left_eye[1] + right_eye[1]) // 2  # 두 눈의 중간 y 좌표

    ears_width = abs(left_eye[0] - right_eye[0]) * 2  # 귀의 너비는 두 눈 사이의 거리의 두 배
    ears_height = int(ears_width * (ears_image.shape[0] / ears_image.shape[1]))  # 귀의 높이 비율 계산
    ears_x = mid_eye_x - (ears_width // 2)  # 귀의 x 좌표 계산
    ears_y = mid_eye_y - ears_height - 50  # 귀의 y 좌표는 고정 (눈의 위치에서 일정 위쪽)

    # 화면을 벗어나지 않도록 귀 위치 제한
    ears_x = max(0, min(ears_x, image.shape[1] - ears_width))
    ears_y = max(0, min(ears_y, image.shape[0] - ears_height))

    ears_resized = cv2.resize(ears_image, (ears_width, ears_height))  # 귀 이미지 크기 조정
    for c in range(0, 3):  # 색 채널에 대해 오버레이 적용 (알파 채널 제외)
        image[ears_y:ears_y + ears_height, ears_x:ears_x + ears_width, c] = \
            image[ears_y:ears_y + ears_height, ears_x:ears_x + ears_width, c] * \
            (1 - ears_resized[:, :, 3] / 255.0) + \
            ears_resized[:, :, c] * (ears_resized[:, :, 3] / 255.0)  # 알파 채널을 이용해 귀 이미지를 결합
    return image



# 말풍선 오버레이 그리기
def draw_speech_bubble(image, bubble_image, landmarks, text="Hello!"):
    nose_top = (landmarks.part(30).x, landmarks.part(30).y)  # 코의 상단 좌표
    bubble_x = nose_top[0] + 20  # 말풍선 x 좌표
    bubble_y = nose_top[1] - 160  # 말풍선 y 좌표

    bubble_resized = cv2.resize(bubble_image, (150, 80))  # 말풍선 이미지 크기 조정
    bubble_height, bubble_width = bubble_resized.shape[:2]

    if bubble_x + bubble_width > image.shape[1] or bubble_y + bubble_height > image.shape[0]:
        return image  # 말풍선이 화면을 벗어나지 않도록 제한

    for c in range(0, 3):  # 색 채널에 대해 오버레이 적용
        image[bubble_y:bubble_y + bubble_height, bubble_x:bubble_x + bubble_width, c] = \
            image[bubble_y:bubble_y + bubble_height, bubble_x:bubble_x + bubble_width, c] * \
            (1 - bubble_resized[:, :, 3] / 255.0) + \
            bubble_resized[:, :, c] * (bubble_resized[:, :, 3] / 255.0)  # 알파 채널을 이용해 말풍선 이미지 결합

    # 말풍선에 텍스트 추가
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = bubble_x + (bubble_width - text_size[0]) // 2  # 텍스트 x 좌표 계산
    text_y = bubble_y + (bubble_height + text_size[1]) // 2  # 텍스트 y 좌표 계산
    cv2.putText(image, text, (bubble_x + 10, bubble_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # 텍스트 삽입
    return image

# 메인 루프
while cap.isOpened():
    ret, image = cap.read()  # 프레임 읽기
    if not ret:
        break

    image = cv2.flip(image, 1)  # 좌우 반전
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    faces = detector(gray)  # 얼굴 탐지

    for face in faces:
        landmarks = predictor(gray, face)  # 얼굴 랜드마크 예측
        face_width = face.right() - face.left()  # 얼굴 너비 계산
        
        if mode in [2, 3, 4]:  # 귀나 말풍선 등 특수 효과 모드
            current_overlay = None
            use_text_overlay = False
            if mode == 2:
                image = draw_ears(image, cat_ears_image, landmarks,face)  # 고양이 귀
            elif mode == 3:
                image = draw_ears(image, rabbit_ears_image, landmarks,face)  # 토끼 귀
            elif mode == 4:
                image = draw_speech_bubble(image, speech_bubble_image, landmarks, text="Hello!")  # 말풍선
        elif mode in [5, 6, 7, 8]:  # 오버레이와 텍스트 오버레이 모드
            if use_text_overlay:
                balloon_text = "why so serious...."  # 텍스트 오버레이
                (text_width, text_height), baseline = cv2.getTextSize(balloon_text, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 2)
                x_offset = face.left() + (face_width - text_width) // 2
                y_offset = face.top() - text_height - 10
                cv2.putText(image, balloon_text, (x_offset, y_offset), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 255), 2)
            elif current_overlay is not None:
                resized_overlay = resize_overlay(face_width, current_overlay)  # 오버레이 크기 조정
                mask = create_mask(resized_overlay)  # 마스크 생성
                overlay_x = face.right() - 10
                overlay_y = face.top() - resized_overlay.shape[0]
                # 오버레이 위치 조정
                if overlay_x + resized_overlay.shape[1] > image.shape[1]:
                    overlay_x = image.shape[1] - resized_overlay.shape[1]
                if overlay_y < 0:
                    overlay_y = 0
                image = overlay_image(image, resized_overlay, overlay_x, overlay_y, mask)  # 오버레이 적용

    # 모드
    instructions = "1:basic/ 2:cat ears/ 3:rabbit ears/ 4:speech bubble/ 5:overlay/ 6:text overlay/ 7:work/ 8:gym"
    cv2.putText(image, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 이미지 출력
    cv2.imshow("Face Detection with Effects", image)

    # 키보드 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == 27: 
        break
    elif key == ord('1'):
        mode = 1  # 기본 모드
    elif key == ord('2'):
        mode = 2  # 고양이 귀
    elif key == ord('3'):
        mode = 3  # 토끼 귀
    elif key == ord('4'):
        mode = 4  # 말풍선
    elif key == ord('5'):
        mode = 5
        current_overlay = handsome_overlay 
        use_text_overlay = False
    elif key == ord('6'): #텍스트
        mode = 6
        use_text_overlay = True 
        current_overlay = None
    elif key == ord('7'):
        mode = 7
        current_overlay = bubble_overlay  
        use_text_overlay = False
    elif key == ord('8'):
        mode = 8
        current_overlay = gym_overlay  
        use_text_overlay = False

# 자원 해제
cap.release()
cv2.destroyAllWindows()
