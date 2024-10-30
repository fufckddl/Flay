import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_al.xml')


#안경을 착용한 눈을 포함해 더 다양한 눈 형태를 감지할 수 있도록 학습
#안경이 있거나 약간 가려진 눈을 감지하는데 좀 더 적합함
#인식 범위 넓지만, 감지 속도가 약간 느리거나, 잘못된 인식률 높을 수 있음

eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

#안경이 없는 눈을 감지하는데 중점을 두고 학습
#안경 착용한 경우나 일부 가려진 경우 정확도 떨어질 수 있음
#상대적으로 더 빠르게 눈을 감지할 수 있지만, 인식범위 제한적
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('face.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y : y + h, x : x + w]
    roi_color = img[y : y + h, x : x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()