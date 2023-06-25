import cv2
import time as t
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
input = cv2.imread('/Users/jperic/Downloads/Face with masks detection.v1i.voc/test/maksssksksss801_png.rf.90a5890962710b3c4262ef693e0f45ae.jpg')
gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY) 
t1 = t.time()
rectangles = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
t2 = t.time()
print(t2-t1)
for (x, y, w, h) in rectangles:
    cv2.rectangle(input, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite("img.jpg", input)
cv2.imshow('detection', input)
cv2.waitKey(0)
cv2.destroyAllWindows()