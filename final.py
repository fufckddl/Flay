import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from utils import apply_filter_mode
from loadImages import cat_ears_image, rabbit_ears_image, speech_bubble_image, handsome_overlay, bubble_overlay, gym_overlay
from loadPredictor import predictor, detector

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)

# tkinter 창 생성 및 설정
root = tk.Tk()
root.title("필터 선택 UI")

# 현재 선택된 필터 및 오버레이 관련 변수를 초기화
mode = tk.IntVar(value=1)  # 기본 필터 모드를 1로 설정
current_overlay = None  # 현재 오버레이 이미지를 저장할 변수
use_text_overlay = tk.BooleanVar(value=False)  # 텍스트 오버레이 여부를 저장하는 변수

# 필터 선택 함수 정의
def select_filter(selected_mode, overlay=None, text_overlay=False):
    global current_overlay
    mode.set(selected_mode)  # 모드 값을 설정
    current_overlay = overlay  # 현재 오버레이 이미지 설정
    use_text_overlay.set(text_overlay)  # 텍스트 오버레이 여부 설정

# 필터 선택 버튼을 생성하여 필터 모드 변경이 가능하도록 구성
button_frame = tk.Frame(root)  # 버튼들을 담을 프레임 생성
button_frame.pack()

# 버튼
buttons = [
    ("normal", 1, None, False),
    ("cat", 2, cat_ears_image, False),
    ("rabbit", 3, rabbit_ears_image, False),
    ("Hello", 4, speech_bubble_image, False),
    ("fish eye", 5, None, False),
    ("잘생긴 사람 첨봐?", 6, handsome_overlay, False),
    ("조커", 7, None, True),
    ("퇴근", 8, bubble_overlay, False),
    ("오운완", 9, gym_overlay, False),
]

# 버튼 생성 및 배치
for (text, btn_mode, overlay, text_overlay) in buttons:
    button = tk.Button(button_frame, text=text, command=lambda m=btn_mode, o=overlay, t=text_overlay: select_filter(m, o, t))
    button.pack(side="left", padx=5, pady=5)  # 버튼을 왼쪽 정렬로 배치, 간격 조절

# 비디오 화면 업데이트 함수
def update_video():
    ret, image = cap.read()
    if not ret:
        return
    
    image = cv2.resize(image, (640, 400))
    image = cv2.flip(image, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        face_width = face.right() - face.left()
        
        # face_rect 매개변수 제거
        image = apply_filter_mode(image, mode.get(), current_overlay, 
                                use_text_overlay.get(), face, landmarks, face_width)

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

# 비디오 업데이트 시작
update_video()

# 창 종료 시 on_closing 함수를 호출하도록 설정
root.protocol("WM_DELETE_WINDOW", on_closing)

# tkinter 메인 루프 실행
root.mainloop()