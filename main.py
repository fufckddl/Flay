import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from background import apply_background_blur, apply_background_with_image
from utils import apply_filter_mode
from loadImages import cat_ears_image, rabbit_ears_image, speech_bubble_image, handsome_overlay, bubble_overlay, gym_overlay
from loadBackground import beach_image
from loadPredictor import predictor, detector
from save import savePic

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)

# tkinter 창 생성 및 설정
root = tk.Tk()
root.title("필터 선택 UI")

# 필터 및 배경 모드 관련 변수를 초기화
filter_mode = tk.IntVar(value=1)  # 기본 필터 모드를 1로 설정
background_mode = tk.IntVar(value=1)  # 기본 배경 모드를 1로 설정

current_overlay = None  # 현재 오버레이 이미지를 저장할 변수
use_text_overlay = tk.BooleanVar(value=False)  # 텍스트 오버레이 여부를 저장하는 변수

# 필터 선택 함수 정의
def select_filter(selected_mode, overlay=None, text_overlay=False):
    global current_overlay
    filter_mode.set(selected_mode)  # 필터 모드 값 설정
    current_overlay = overlay  # 현재 오버레이 이미지 설정
    use_text_overlay.set(text_overlay)  # 텍스트 오버레이 여부 설정

# 배경 선택 함수 정의
def select_background(selected_bgmode):
    background_mode.set(selected_bgmode)  # 배경 모드 값 설정

# 필터 버튼 프레임 생성 및 배치
filter_button_frame = tk.Frame(root)  
filter_button_frame.pack()

# 배경 버튼 프레임 생성 및 배치
background_button_frame = tk.Frame(root)  
background_button_frame.pack()

# 필터 버튼 리스트
filter_buttons = [
    ("normal", 1, None, False),
    ("cat", 2, cat_ears_image, False),
    ("rabbit", 3, rabbit_ears_image, False),
    ("Hello", 4, speech_bubble_image, False),
    ("fish eye", 5, None, False),
    ("잘생긴 사람 첨봐?", 6, handsome_overlay, False),
    ("조커", 7, None, True),
    ("퇴근", 8, bubble_overlay, False),
    ("오운완", 9, gym_overlay, False)
]

# 배경 버튼 리스트
bg_buttons = [
    ('배경: normal', 1),
    ('배경: 블러', 2),
    ('배경: 바다', 3)
]

# 필터 버튼 생성 및 배치
for (text, btn_mode, overlay, text_overlay) in filter_buttons:
    button = tk.Button(filter_button_frame, text=text, command=lambda m=btn_mode, o=overlay, t=text_overlay: select_filter(m, o, t))
    button.pack(side="left", padx=5, pady=5)  # 버튼을 왼쪽 정렬로 배치, 간격 조절

# 배경 버튼 생성 및 배치
for (text, btn_bgmode) in bg_buttons:
    button = tk.Button(background_button_frame, text=text, command=lambda m=btn_bgmode: select_background(m))
    button.pack(side="left", padx=5, pady=5)  # 버튼을 왼쪽 정렬로 배치, 간격 조절

# 전역 변수로 얼굴 정보 저장
current_face = None
current_landmarks = None
current_face_width = 0

# 비디오 화면 업데이트 함수
def update_video():
    global current_face, current_landmarks, current_face_width
    ret, image = cap.read()  # 현재 비디오 프레임을 읽기
    if not ret:
        return

    # 비디오 프레임 크기 조정
    image = cv2.resize(image, (640, 400))
    # 배경 블러가 선택된 경우 apply_background_blur 함수 호출
    if background_mode.get() == 2:
        image = apply_background_blur(image)

    if background_mode.get() == 3:
        image = apply_background_with_image(image, beach_image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 얼굴 탐지에 사용할 그레이스케일 이미지 생성
    faces = detector(gray)  # 얼굴 탐지

    # 필터를 적용할 때 필요한 얼굴 정보
    current_face = None
    current_landmarks = None
    current_face_width = 0

    # 탐지된 얼굴 각각에 대해 필터 적용
    for face in faces:
        current_landmarks = predictor(gray, face)  # 얼굴 랜드마크 예측
        current_face_width = face.right() - face.left()  # 얼굴 너비 계산
        # 첫 번째 얼굴만 사용할 경우 current_face = face[0]로 설정할 수 있음
        current_face = face
        break

    # 필터 적용 함수 호출하여 선택된 모드와 오버레이를 적용
    if current_face is not None:
        image = apply_filter_mode(image, filter_mode.get(), current_overlay, use_text_overlay.get(), current_face, current_landmarks, current_face_width)

    # OpenCV 이미지를 tkinter에서 표시할 수 있도록 변환
    cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
    img = Image.fromarray(cv2image)  # PIL 이미지로 변환
    imgtk = ImageTk.PhotoImage(image=img)  # tkinter용 이미지로 변환
    video_label.imgtk = imgtk  # 레퍼런스 유지 (GC 방지)
    video_label.configure(image=imgtk)  # 레이블에 이미지 설정

    # 주기적으로 업데이트 (20ms마다 update_video 호출)
    video_label.after(20, update_video)


# 비디오 화면을 표시할 레이블 생성
video_label = tk.Label(root)
video_label.pack()

# 종료 버튼 생성
def on_closing():
    cap.release()  # 비디오 캡처 해제
    root.destroy()  # tkinter 창 닫기

# 종료 버튼 생성 및 배치
exit_button = tk.Button(root, text="종료", command=on_closing)
exit_button.pack()

# 촬영 버튼 클릭 시 호출되는 함수
save_button = tk.Button(root, text="촬영", command=lambda: savePic(apply_filter_mode(cap.read()[1], filter_mode.get(), current_overlay, use_text_overlay.get(), current_face, current_landmarks, current_face_width), './savePictures') if cap.read()[0] else print("Error: Could not capture frame."))
save_button.pack()


# 비디오 업데이트 시작
update_video()

# 창 종료 시 on_closing 함수를 호출하도록 설정
root.protocol("WM_DELETE_WINDOW", on_closing)

# tkinter 메인 루프 실행
root.mainloop()
