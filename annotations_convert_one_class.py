import os
import xml.etree.ElementTree as ET

def modify_annotation(input_dir, output_dir):

    for filename in os.listdir(input_dir):
        if filename.endswith('.xml'): 
            tree = ET.parse(os.path.join(input_dir, filename))
            root = tree.getroot()

            for obj in root.iter('object'):
                name_tag = obj.find('name')

                name_tag.text = 'face'

            tree.write(os.path.join(output_dir, filename))

input_dir = '/Users/jperic/Downloads/archive/annotations'
output_dir = '/Users/jperic/Downloads/archive/face_annotations'

os.makedirs(output_dir, exist_ok=True)

modify_annotation(input_dir, output_dir)


