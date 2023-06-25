import cv2
import numpy as np
import xml.etree.ElementTree as ET
import dlib
import os
import tensorflow as tf
import json
from keras.models import load_model

from object_detection.builders import model_builder
from object_detection.utils import config_util

# Konfiguracija i model za SSD
configs = config_util.get_configs_from_pipeline_file('/Users/jperic/Documents/private/rusu_projekt/my_ssd_mobnet_faces/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('/Users/jperic/Documents/private/rusu_projekt/my_ssd_mobnet_faces', 'ckpt-11')).expect_partial()

vgg = load_model('vgg.h5')
mobnet = load_model('mobnetv2.h5')
my_model = load_model('my_model.h5')

# TensorFlow funkcija za detekciju
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Ostali detektori
detector = dlib.get_frontal_face_detector()
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_objects_hog_svm(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections, _, scores = detector.run(img, 0, -1)

    boxes = []
    for d, score in zip(detections, scores):
        left = max(0, d.left())
        top = max(0, d.top())
        right = min(image.shape[1], d.right())
        bottom = min(image.shape[0], d.bottom())
        width = right - left
        height = bottom - top
        boxes.append((left, top, width, height, score))

    return boxes


def detect_object_ssd(image):
    height, width, _ = image.shape
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()

    boxes[:, [0, 2]] *= height  
    boxes[:, [1, 3]] *= width 

    rectangles = np.zeros_like(boxes)
    rectangles[:, 0] = boxes[:, 1]
    rectangles[:, 1] = boxes[:, 0]
    rectangles[:, 2] = boxes[:, 3] - boxes[:, 1]
    rectangles[:, 3] = boxes[:, 2] - boxes[:, 0]

    scores = scores.reshape(-1, 1)
    
    return np.hstack((rectangles, scores))

def detect_objects_haar(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    
    detections, weights = cascade.detectMultiScale2(img, scaleFactor=1.1, minNeighbors=4)
    return [(x, y, w, h, weight) for (x, y, w, h), weight in zip(detections, weights)]

def calculate_area(bbox):
    return bbox[2] * bbox[3]

def bbox_to_segmentation(bbox):
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = x_min + bbox[2]  
    y_max = y_min + bbox[3]  
    return [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]]

def main(image_folder, method='ssd', output_json='detections.json', model_name = "vgg"):
    if model_name == 'vgg':
        model = vgg
    elif model_name == 'mobnet':
        model = mobnet
    elif model_name == 'my_model':
        model = my_model
    else:
        raise ValueError(f"Unknown model_name {model_name}")
    num = 0
    json_output = []

    filenames = sorted(os.listdir(image_folder))
    detection_idx=0
    for filename in filenames:
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if method == 'hog_svm':
                detections = detect_objects_hog_svm(image)
            elif method == 'haar':
                detections = detect_objects_haar(image)
            elif method == 'ssd':
                detections = detect_object_ssd(image)
            else:
                raise ValueError(f"Unknown method {method}")
            
            for detection in detections:
                detection_rounded = np.round(detection).astype(int) 
                cropped = image[detection_rounded[1]:(detection_rounded[1] + detection_rounded[3]),detection_rounded[0]:(detection_rounded[0] + detection_rounded[2])]
                cropped = cv2.resize(cropped, (128, 128))
                cropped = cropped.reshape(1, 128, 128, 3)
                prediction = model.predict(cropped, verbose=0)
                class_id = np.argmax(prediction)
                json_output.append({
                    "id": int(detection_idx + 1),
                    "image_id": int(num +1),
                    "area": float(calculate_area(detection[:4])),
                    "category_id": int(class_id + 1),
                    "bbox": [float(i) for i in detection[:4]],
                    "segmentation": [[float(i) for i in seg] for seg in bbox_to_segmentation(detection[:4])],
                    "iscrowd": int(0),
                    "score": float(detection[4])
                })
                detection_idx+=1
            num += 1
    with open(output_json, 'w') as f:
        json.dump(json_output, f)

main('/Users/jperic/Downloads/Mask detection.v2i.voc/test', 'haar','eval/results/haar_and_vgg_detections.json', "vgg")
