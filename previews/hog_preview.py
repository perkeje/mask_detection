import cv2
import dlib
import time
detector = dlib.get_frontal_face_detector()
input = cv2.imread('/Users/jperic/Downloads/Face with masks detection.v1i.voc/test/maksssksksss832_png.rf.edc273699d8145da69523db2f7037b97.jpg')
gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY) 
t1 = time.time()
rectangles = detector(gray)
t2 = time.time()
print(t2-t1)
for rect in rectangles:
    x = rect.left()
    y = rect.top()
    w = rect.width()
    h = rect.height()
    cv2.rectangle(input, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite("img_hog.jpg", input)
cv2.imshow('detection', input)
cv2.waitKey(0)
cv2.destroyAllWindows()