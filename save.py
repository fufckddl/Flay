import cv2
from datetime import datetime

def savePic(frame, save_dir="."):
    # 현재 날짜와 시간을 기반으로 고유 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/capture_{timestamp}.jpg"
    
    # 이미지 저장
    cv2.imwrite(filename, frame)
    print(f"Image saved as {filename}")
