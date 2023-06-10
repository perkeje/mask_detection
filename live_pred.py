import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K


class_dict = {0:"Mask Incorrectly", 1:"Mask", 2:"No Mask"}

model = load_model('mobnetv2.h5')

cap = cv2.VideoCapture(0)

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rectangles = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in rectangles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img = gray[y:y+h, x:x+w]
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.reshape(1, 128, 128, 3)

        prediction = model.predict(img, verbose=0)
        class_id = np.argmax(prediction)

        class_label = class_dict[class_id]
        class_prob = prediction[0][class_id]

        text = f"{class_label}: {class_prob:.2f}"

        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
