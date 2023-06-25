import cv2
import numpy as np
import os
import tensorflow as tf
import json
import time

from object_detection.builders import model_builder
from object_detection.utils import config_util

# Konfiguracija i model za SSD
configs = config_util.get_configs_from_pipeline_file('/Users/jperic/Documents/private/rusu_projekt/my_ssd_mobnet/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('/Users/jperic/Documents/private/rusu_projekt/my_ssd_mobnet', 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def detect_object_ssd(image):
    height, width, _ = image.shape
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy()
    
    boxes[:, [0, 2]] *= height  
    boxes[:, [1, 3]] *= width 

    rectangles = np.zeros_like(boxes)
    rectangles[:, 0] = boxes[:, 1]
    rectangles[:, 1] = boxes[:, 0]
    rectangles[:, 2] = boxes[:, 3] - boxes[:, 1]
    rectangles[:, 3] = boxes[:, 2] - boxes[:, 0]

    scores = scores.reshape(-1, 1)
    class_indices = classes.reshape(-1, 1)
    
    return np.hstack((rectangles, scores,class_indices))


def calculate_area(bbox):
    return bbox[2] * bbox[3]

def bbox_to_segmentation(bbox):
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = x_min + bbox[2]  
    y_max = y_min + bbox[3]  
    return [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]]

def main(image_folder, output_json='detections.json'):
    num = 0
    json_output = []
    times = []
    filenames = sorted(os.listdir(image_folder))
    detection_idx=0
    for filename in filenames:
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            time1 = time.time()
            detections = detect_object_ssd(image)
            time_curr = time.time() - time1
            time_sum = 0
            for detection in detections:
                json_output.append({
                    "id": int(detection_idx + 1),
                    "image_id": int(num +1),
                    "area": float(calculate_area(detection[:4])),
                    "category_id": int(detection[5] + 1),
                    "bbox": [float(i) for i in detection[:4]],
                    "segmentation": [[float(i) for i in seg] for seg in bbox_to_segmentation(detection[:4])],
                    "iscrowd": int(0),
                    "score": float(detection[4])
                })
                detection_idx+=1
            num += 1
            time_accumulated = time_sum + time_curr
            times.append(time_accumulated)
    print(f'Average detection and classification time: {np.mean(times)} seconds')
    with open(output_json, 'w') as f:
        json.dump(json_output, f)

main('/Users/jperic/Downloads/Mask detection.v2i.voc/test','eval/results/ssd_complete.json')
