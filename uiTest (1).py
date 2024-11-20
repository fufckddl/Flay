import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QComboBox, QHBoxLayout, QWidget, QMenuBar
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
from background import apply_background_blur, apply_background_with_image
from loadImages import cat_ears_image, rabbit_ears_image, speech_bubble_image, handsome_overlay, bubble_overlay, gym_overlay
from loadBackground import beach_image
from loadPredictor import predictor, detector
from save import savePic
from utils import apply_filter_mode


class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # 비디오 캡처 초기화
        self.cap = cv2.VideoCapture(1)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        # 설정 초기화
        self.filter_mode = 1
        self.background_mode = 1
        self.current_overlay = None
        self.use_text_overlay = False

    def init_ui(self):
        self.setWindowTitle("필터카메라")
        self.setGeometry(100, 100, 800, 600)

        # 메인 위젯 및 레이아웃 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # 상단바 레이아웃
        top_bar = QHBoxLayout()

        # 종료 버튼 (왼쪽 정렬)
        exit_button = QPushButton("종료")
        exit_button.setStyleSheet("""
                    QPushButton {
                        background-color: white; /* 버튼 배경을 흰색으로 설정 */
                        color: black; /* 글씨를 검은색으로 설정 */
                        border: 1px solid gray; /* 테두리를 회색으로 설정 */
                        padding: 5px;
                    }
                    QPushButton:hover {
                        background-color: lightgray; /* 마우스를 올리면 버튼 배경을 연회색으로 변경 */
                    }
                """)
        exit_button.clicked.connect(self.close)
        top_bar.addWidget(exit_button, alignment=Qt.AlignLeft)

        # 드롭다운 (오른쪽 정렬)
        self.bg_dropdown = QComboBox(self)
        self.bg_dropdown.addItems(["Normal", "Blur", "Beach"])
        self.bg_dropdown.setStyleSheet("""
            QComboBox {
                background-color: white;
                color: black;
                border: 1px solid gray;
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
                selection-background-color: lightgray;
                selection-color: black;
            }
        """)
        self.bg_dropdown.currentIndexChanged.connect(self.select_background)
        self.bg_dropdown.setFixedWidth(120)
        top_bar.addWidget(self.bg_dropdown, alignment=Qt.AlignRight)

        # 상단바를 레이아웃에 추가
        layout.addLayout(top_bar)

        # 실시간 영상 레이블
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # 필터 버튼들
        self.filter_layout = QHBoxLayout()
        filters = [
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
        for text, mode, overlay, text_overlay in filters:
            btn = QPushButton(text)
            btn.clicked.connect(lambda _, m=mode, o=overlay, t=text_overlay: self.select_filter(m, o, t))
            self.filter_layout.addWidget(btn)

        layout.addLayout(self.filter_layout)

        # 캡처 버튼
        self.capture_button = QPushButton("📸")
        self.capture_button.setStyleSheet("background-color: lightblue;")
        self.capture_button.clicked.connect(self.capture_image)
        layout.addWidget(self.capture_button, alignment=Qt.AlignCenter)

    def select_filter(self, mode, overlay, text_overlay):
        self.filter_mode = mode
        self.current_overlay = overlay
        self.use_text_overlay = text_overlay

    def select_background(self, index):
        self.background_mode = index + 1

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))

        if self.background_mode == 2:
            frame = apply_background_blur(frame)
        elif self.background_mode == 3:
            frame = apply_background_with_image(frame, beach_image)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            face_width = face.right() - face.left()
            frame = apply_filter_mode(frame, self.filter_mode, self.current_overlay, self.use_text_overlay, face, landmarks, face_width)
            break

        # OpenCV 이미지를 PyQt5에서 표시할 수 있도록 변환
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

    def capture_image(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            return

        # 배경 필터 적용
        if self.background_mode == 2:
            frame = apply_background_blur(frame)
        elif self.background_mode == 3:
            frame = apply_background_with_image(frame, beach_image)

        # 얼굴 탐지 및 필터 적용
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            face_width = face.right() - face.left()
            frame = apply_filter_mode(frame, self.filter_mode, self.current_overlay, self.use_text_overlay, face,
                                      landmarks, face_width)
            break

        # 필터가 적용된 이미지를 저장
        savePic(frame, "./savePictures")

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
