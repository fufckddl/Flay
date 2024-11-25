from loadPredictor import segmentation
import cv2
import numpy as np

def apply_background_blur(frame):
    """
    사람 영역을 제외한 나머지 배경에 블러 효과를 적용하여 반환합니다.
    
    Parameters:
        frame (numpy.ndarray): 현재 카메라 프레임.
        
    Returns:
        numpy.ndarray: 사람이 포함된 영역만 블러 처리된 프레임.
    """
    # MediaPipe로 사람과 배경 분할
    results = segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #selfie segmentation API 사용해서 전경과 배경을 분리하여 
    
    # 마스크 생성 (사람 부분이 255인 이진화된 마스크)
    mask = results.segmentation_mask 
    mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    
    # 블러 처리: 얼굴과 몸을 제외한 배경 부분에만 블러 적용
    blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
    combined_frame = np.where(mask[..., None] == 255, frame, blurred_frame)

    return combined_frame

def apply_background_with_image(frame, background_image):
    """
    사람 영역을 제외한 나머지 배경에 이미지를 덮어씌워 반환합니다.
    
    Parameters:
        frame (numpy.ndarray): 현재 카메라 프레임.
        background_image (numpy.ndarray): 배경으로 사용할 이미지.
        
    Returns:
        numpy.ndarray: 사람이 포함된 영역 외에 배경 이미지가 적용된 프레임.
    """
    # 배경 이미지 크기 조정
    background_image = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
    
    # MediaPipe로 사람과 배경 분할
    results = segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 마스크 생성 (사람 부분이 255인 이진화된 마스크)
    mask = results.segmentation_mask
    mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    
    # 사람 영역만 유지하고, 나머지 배경에 이미지 적용
    combined_frame = np.where(mask[..., None] == 255, frame, background_image)

    return combined_frame