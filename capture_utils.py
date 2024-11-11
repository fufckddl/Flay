import cv2

def initialize_capture():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Face Detection with Filters')
    brightness_trackbar = cv2.createTrackbar('Brightness', 'Face Detection with Filters', 50, 100, lambda x: x)
    contrast_trackbar = cv2.createTrackbar('Contrast', 'Face Detection with Filters', 50, 100, lambda x: x)
    return cap, brightness_trackbar, contrast_trackbar

def get_adjusted_frame(frame, brightness_trackbar, contrast_trackbar):
    brightness = cv2.getTrackbarPos('Brightness', 'Face Detection with Filters') - 50
    contrast = cv2.getTrackbarPos('Contrast', 'Face Detection with Filters') / 50.0
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    return adjusted_frame
