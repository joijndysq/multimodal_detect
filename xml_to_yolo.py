#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ['person']

def convert_voc_to_yolo(xml_path, img_width, img_height):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    yolo_labels = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASSES:
            continue
        
        class_id = CLASSES.index(name)
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_labels

def main():
    base_dir = Path('dataset/LLVIP')
    annotations_dir = base_dir / 'Annotations'
    output_dir = Path('dataset/yolo/labels')
    
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    train_prefixes = {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                     '11', '12', '13', '14', '15', '16', '17', '18', '25'}
    test_prefixes = {'19', '20', '21', '22', '23', '24', '26'}
    
    train_count = 0
    test_count = 0
    
    for xml_file in sorted(annotations_dir.glob('*.xml')):
        stem = xml_file.stem
        
        prefix = stem[:2]
        
        if prefix in train_prefixes:
            out_dir = train_dir
            train_count += 1
        elif prefix in test_prefixes:
            out_dir = test_dir
            test_count += 1
        else:
            continue
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        yolo_labels = convert_voc_to_yolo(xml_file, img_width, img_height)
        
        if yolo_labels:
            label_file = out_dir / f"{stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_labels))
    
    print(f"Train: {train_count} labels")
    print(f"Test: {test_count} labels")

if __name__ == '__main__':
    main()