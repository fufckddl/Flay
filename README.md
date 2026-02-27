# Flay

**실시간 필터 카메라** — OpenCV, dlib, MediaPipe, PyQt5 기반 데스크톱 앱. 웹캠 영상에 얼굴 필터·배경 효과를 적용하고 캡처할 수 있습니다.

---

## 주요 기능

- **실시간 얼굴 필터**
  - **Normal** — 필터 없음
  - **Cat** — 고양이 귀·코 오버레이
  - **Bear** — 곰 귀·코 오버레이
  - **Fish eye** — 얼굴 중심 피쉬아이 왜곡
  - **Nice!** — 말풍선 스타일 오버레이
  - **Work out** — 헬스/운동 테마 오버레이

- **배경 모드**
  - **Normal** — 원본 배경
  - **Blur** — 사람 제외 배경 블러 (MediaPipe 세그멘테이션)
  - **Beach** — 해변 이미지로 가상 배경 교체

- **조절**
  - 밝기·명암 슬라이더
  - 📸 버튼으로 현재 화면 캡처 → `savePictures/` 폴더에 저장

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| GUI | PyQt5 |
| 영상 처리 | OpenCV (cv2) |
| 얼굴 검출·랜드마크 | dlib (68점 랜드마크) |
| 배경 분할 | MediaPipe Selfie Segmentation |
| 언어 | Python 3 |

---

## 프로젝트 구조

```
Flay/
├── main.py           # 앱 진입점, PyQt5 윈도우·캠·UI 이벤트
├── loadPredictor.py  # dlib 얼굴 검출기/예측기, MediaPipe 세그멘테이션 초기화
├── loadImages.py     # 필터용 이미지 로드 (귀, 코, 말풍선, 오버레이 등)
├── loadBackground.py # 가상 배경 이미지 로드 (beach.jpg)
├── background.py     # 배경 블러·가상 배경 적용 (MediaPipe 마스크 활용)
├── overlays.py       # 귀/코/말풍선 오버레이, 피쉬아이 필터, 랜드마크 스무딩
├── utils.py          # 필터 모드별 적용 로직 (apply_filter_mode 등)
├── save.py           # 캡처 이미지 저장 (타임스탬프 파일명)
├── images/           # 필터·배경용 이미지 (필수)
│   ├── cat_ears.png, cat_nose.png
│   ├── bear_ears.png, bear_nose.png
│   ├── speech_bubble.png, nice.png, gym.png
│   ├── handsome.png (선택)
│   └── beach.jpg     # 가상 배경
├── savePictures/     # 캡처 저장 폴더 (실행 시 생성 가능)
└── shape_predictor_68_face_landmarks.dat  # dlib 랜드마크 모델 (별도 다운로드)
```

---

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install opencv-python PyQt5 dlib mediapipe numpy
```

### 2. dlib 랜드마크 모델

프로젝트 루트에 `shape_predictor_68_face_landmarks.dat` 파일이 있어야 합니다.

- 다운로드: [dlib 68 face landmarks](https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat) 등 공개된 68점 모델 사용

### 3. 이미지 에셋

`./images/` 폴더를 만들고 아래 파일을 넣습니다.

| 파일 | 용도 |
|------|------|
| `cat_ears.png`, `cat_nose.png` | 고양이 필터 |
| `bear_ears.png`, `bear_nose.png` | 곰 필터 |
| `speech_bubble.png` | 말풍선 (선택) |
| `nice.png` | Nice! 오버레이 |
| `gym.png` | Work out 오버레이 |
| `beach.jpg` | 가상 배경 (Beach 모드) |

### 4. 실행

```bash
python main.py
```

- 웹캠은 기본적으로 `cv2.VideoCapture(1)` 사용. 카메라 인덱스가 다르면 `main.py` 19번 줄에서 `0` 등으로 변경하세요.

---

## 요약

- **Flay**는 웹캠 입력에 실시간으로 얼굴 필터(귀/코/피쉬아이/오버레이)와 배경 효과(블러/가상 배경)를 적용하는 **Python + OpenCV + PyQt5** 필터 카메라 앱입니다.
- 얼굴 검출·랜드마크는 **dlib**, 배경 분할은 **MediaPipe**로 처리하며, 캡처한 화면은 `savePictures/`에 저장됩니다.
