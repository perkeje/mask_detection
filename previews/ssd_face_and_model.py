import cv2
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

configs = config_util.get_configs_from_pipeline_file('/Users/jperic/Documents/private/rusu_projekt/my_ssd_mobnet_faces/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('/Users/jperic/Documents/private/rusu_projekt/my_ssd_mobnet_faces', 'ckpt-11')).expect_partial()

class_dict = {0:"Mask Incorrectly", 1:"Mask", 2:"No Mask"}

# Load the model with custom metrics
model = load_model('my_model.h5')

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:

    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor) 


    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()

   
    confidence_threshold = 0.5  
    boxes = boxes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]
    boxes[:, [0, 2]] *= height  
    boxes[:, [1, 3]] *= width 

    rectangles = np.zeros_like(boxes)
    rectangles[:, 0] = boxes[:, 1]  
    rectangles[:, 1] = boxes[:, 0]  
    rectangles[:, 2] = boxes[:, 3] - boxes[:, 1]  
    rectangles[:, 3] = boxes[:, 2] - boxes[:, 0]  
    rectangles = rectangles.astype(int)

    for (x, y, w, h) in rectangles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img = frame[y:y+h, x:x+w]
        img = cv2.resize(img, (128, 128)) 
        

        img = img.reshape(1, 128, 128, 3)

        prediction = model.predict(img, verbose=0)
        class_id = np.argmax(prediction)

        class_label = class_dict[class_id]
        class_prob = prediction[0][class_id]

        text = f"{class_label}: {class_prob:.2f}"

        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)


    cv2.imshow('Video', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
