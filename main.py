import cv2
import numpy as np

video_cap = cv2.VideoCapture(0)

background_subtractor = cv2.createBackgroundSubtractorKNN()

kernel = np.ones(shape = (5,5), dtype = np.uint8)

while True:
    ret, frame = video_cap.read()

    if not ret:
        break

    mask = background_subtractor.apply(image = frame)

    mask = cv2.erode(src = mask, kernel=kernel)

    mask = cv2.findNonZero(src = mask)

    x1 ,y1, x2, y2 = cv2.boundingRect(array = mask)

    if mask is not None:
        cv2.rectangle(img = frame, pt1 = (x1,y1), pt2 = (x1+x2, y1+y2), color = (0,0,0), thickness = 4, lineType = cv2.LINE_AA)
    
    cv2.imshow(winname="Captured Video", mat=cv2.resize(src = frame, dsize = None, fx = 1.3, fy = 1.3 ))

    key = cv2.waitKey(delay = 1)

    if key == ord("q") or key == ord("Q"):
        break

video_cap.release()