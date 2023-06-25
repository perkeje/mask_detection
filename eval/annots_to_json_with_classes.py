import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import json

class_dict = {"mask_weared_incorrect":1, "with_mask":2, "without_mask":3}

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for object in root.findall('object'):
        name = object.find('name').text
        bndbox = object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append((class_dict[name],[xmin, ymin, xmax-xmin, ymax-ymin]))
    return objects

def load_images_and_annotations(folder):
    files = sorted(os.listdir(folder))
    for filename in files:
        if filename.endswith(".xml"):
            xml_path = os.path.join(folder, filename)
            image_path = os.path.join(folder, filename.replace('.xml', '.jpg'))
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                annotations = parse_xml(xml_path)
                yield image, annotations, image_path

def calculate_area(bbox):
    return bbox[2] * bbox[3]

def bbox_to_segmentation(bbox):
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = x_min + bbox[2]  
    y_max = y_min + bbox[3] 
    return [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]]

def main(image_folder, output_json='annotations_test.json'):
    num = 0
    json_output = {
        "licenses": [],
        "info": {
            "contributor": "",
            "description": "",
            "url": "",
            "date_created": "",
            "version": "",
            "year": 0
        },
        "annotations": [],
        "images": [],
        "categories": []
    }
    annotation_idx=0
    for image, annotations, image_path in load_images_and_annotations(image_folder):
        for annotation in annotations:
            json_output["annotations"].append({
                "id": annotation_idx + 1,
                "image_id": num + 1,
                "area": float(calculate_area(annotation[1])),
                "category_id": annotation[0],
                "bbox": [float(i) for i in annotation[1]],
                "segmentation": [[float(i) for i in seg] for seg in bbox_to_segmentation(annotation[1])],
                "iscrowd": False
            })
            annotation_idx+=1

        num += 1

    num = 0
    for image, annotations, image_path in load_images_and_annotations(image_folder):
        height, width, _ = image.shape

        json_output["images"].append({
            "width": width,
            "height": height,
            "flickr_url": "",
            "coco_url": "",
            "file_name": os.path.basename(image_path),
            "date_captured": 0,
            "license": 0,
            "id": num + 1
        })
        num += 1

    json_output["categories"].append({
        
        "id": 1,
        "name": "mask_weared_incorrect",
        "supercategory": ""
    
    })
    json_output["categories"].append({
        
        "id": 2,
        "name": "with_mask",
        "supercategory": ""
    
    })
    json_output["categories"].append({
        
        "id": 3,
        "name": "without_mask",
        "supercategory": ""
    
    })
    
    with open(output_json, 'w') as f:
        json.dump(json_output, f)

main('/Users/jperic/Downloads/Mask detection.v2i.voc/test', 'eval/results/annotations_test_with_classes.json')
