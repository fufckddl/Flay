import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QComboBox, QHBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from background import apply_background_blur, apply_background_with_image
from loadImages import cat_ears_image, rabbit_ears_image, speech_bubble_image, handsome_overlay, bubble_overlay, gym_overlay
from loadBackground import beach_image
from loadPredictor import predictor, detector
from save import savePic
from utils import apply_filter_mode


class VideoProcessor(QThread):
    frame_processed = pyqtSignal(QImage)

    def __init__(self, cap, filter_mode, background_mode, current_overlay, use_text_overlay):
        super().__init__()
        self.cap = cap
        self.filter_mode = filter_mode
        self.background_mode = background_mode
        self.current_overlay = current_overlay
        self.use_text_overlay = use_text_overlay
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (640, 480))

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
                frame = apply_filter_mode(
                    frame,
                    self.filter_mode,
                    self.current_overlay,
                    self.use_text_overlay,
                    face,
                    landmarks,
                    face_width,
                )
                break

            # OpenCV 이미지를 QImage로 변환
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_processed.emit(qt_image)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # 비디오 캡처 초기화
        self.cap = cv2.VideoCapture(0)

        # 설정 초기화
        self.filter_mode = 1
        self.background_mode = 1
        self.current_overlay = None
        self.use_text_overlay = False

        # 작업 스레드 초기화
        self.video_thread = VideoProcessor(
            self.cap, self.filter_mode, self.background_mode, self.current_overlay, self.use_text_overlay
        )
        self.video_thread.frame_processed.connect(self.update_frame)
        self.video_thread.start()

    def init_ui(self):
        self.setWindowTitle("필터카메라")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # 상단 바
        top_bar = QHBoxLayout()
        exit_button = QPushButton("종료")
        exit_button.clicked.connect(self.close)
        top_bar.addWidget(exit_button, alignment=Qt.AlignLeft)

        self.bg_dropdown = QComboBox()
        self.bg_dropdown.addItems(["Normal", "Blur", "Beach"])
        self.bg_dropdown.currentIndexChanged.connect(self.select_background)
        top_bar.addWidget(self.bg_dropdown, alignment=Qt.AlignRight)

        layout.addLayout(top_bar)

        # 영상 출력
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # 필터 버튼
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
            ("오운완", 9, gym_overlay, False),
        ]
        for text, mode, overlay, text_overlay in filters:
            btn = QPushButton(text)
            btn.clicked.connect(lambda _, m=mode, o=overlay, t=text_overlay: self.select_filter(m, o, t))
            self.filter_layout.addWidget(btn)
        layout.addLayout(self.filter_layout)

    def select_filter(self, mode, overlay, text_overlay):
        self.filter_mode = mode
        self.current_overlay = overlay
        self.use_text_overlay = text_overlay
        self.video_thread.filter_mode = mode
        self.video_thread.current_overlay = overlay
        self.video_thread.use_text_overlay = text_overlay

    def select_background(self, index):
        self.background_mode = index + 1
        self.video_thread.background_mode = self.background_mode

    def update_frame(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.video_thread.stop()
        self.cap.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
