import cv2
import numpy as np
import tensorflow as tf
import dlib
from keras.models import load_model
import time
import os
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

def detect_and_process(image, confidence_threshold=0.5):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()

    boxes = boxes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]
    boxes[:, [0, 2]] *= image.shape[1] 
    boxes[:, [1, 3]] *= image.shape[0]

    rectangles = np.zeros_like(boxes)
    rectangles[:, 0] = boxes[:, 1]  
    rectangles[:, 1] = boxes[:, 0]  
    rectangles[:, 2] = boxes[:, 3] - boxes[:, 1]  
    rectangles[:, 3] = boxes[:, 2] - boxes[:, 0]  
    rectangles = rectangles.astype(int)
    
    return rectangles, scores


def detect_ssd(image, confidence_threshold=0.5):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy()

    valid_indices = scores >= confidence_threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    classes = classes[valid_indices]

    boxes[:, [0, 2]] *= image.shape[1]
    boxes[:, [1, 3]] *= image.shape[0]

    rectangles = np.zeros_like(boxes)
    rectangles[:, 0] = boxes[:, 1]
    rectangles[:, 1] = boxes[:, 0]
    rectangles[:, 2] = boxes[:, 3] - boxes[:, 1]
    rectangles[:, 3] = boxes[:, 2] - boxes[:, 0]
    rectangles = rectangles.astype(int)

    return rectangles, scores, classes


configs = config_util.get_configs_from_pipeline_file('/Users/jperic/Documents/private/rusu_projekt/my_ssd_mobnet/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('/Users/jperic/Documents/private/rusu_projekt/my_ssd_mobnet', 'ckpt-11')).expect_partial()

configs_faces = config_util.get_configs_from_pipeline_file('/Users/jperic/Documents/private/rusu_projekt/my_ssd_mobnet_faces/pipeline.config')
detection_model_faces = model_builder.build(model_config=configs['model'], is_training=False)

ckpt2 = tf.compat.v2.train.Checkpoint(model=detection_model_faces)
ckpt2.restore(os.path.join('/Users/jperic/Documents/private/rusu_projekt/my_ssd_mobnet_faces', 'ckpt-11')).expect_partial()

detector = dlib.get_frontal_face_detector()
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = load_model('vgg.h5')

def detect_objects_hog_svm(image):
    detections, _, scores = detector.run(image, 0, -1)

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

def detect_objects_haar(image):
    detections = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4)
    return [(x, y, w, h, 1) for (x, y, w, h) in detections]


def measure_detection_time(image_folder):
    filenames = [f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]

    total_time = 0
    for filename in filenames:
        image = cv2.imread(os.path.join(image_folder, filename))
        start_time = time.time()
        # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detections = detect_objects_hog_svm(img_gray)
        # detections = detect_objects_haar(img_gray)
        # detections,scores = detect_and_process(img_color)
        detections = detect_ssd(img_color)
        # for rect in detections:
        #     x = rect[0]
        #     y = rect[1]
        #     w = rect[2]
        #     h = rect[3]

        #     img = img_color[y:y+h, x:x+w]
        #     img = cv2.resize(img, (128, 128))

        #     img = img.reshape(1, 128, 128, 3)

        #     prediction = model.predict(img, verbose=0)
        total_time += time.time() - start_time

    average_time = total_time / len(filenames) if filenames else 0

    print(f"Average detection time: {average_time} seconds.")

measure_detection_time('/Users/jperic/Downloads/Mask detection.v2i.voc/test')
