#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

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
    base_dir = Path('/home/lyric/ir_rgb/dataset/LLVIP')
    output_dir = Path('/home/lyric/ir_rgb/dataset/yolo')
    
    annotations_dir = base_dir / 'Annotations'
    splits = {'train': base_dir / 'infrared' / 'train', 'test': base_dir / 'infrared' / 'test'}
    
    for split, img_dir in splits.items():
        label_dir = output_dir / 'labels' / split
        label_dir.mkdir(parents=True, exist_ok=True)
        
        image_link_dir = output_dir / 'images' / split
        image_link_dir.mkdir(parents=True, exist_ok=True)
        
        for xml_file in sorted(annotations_dir.glob('*.xml')):
            stem = xml_file.stem
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            yolo_labels = convert_voc_to_yolo(xml_file, img_width, img_height)
            
            if yolo_labels:
                label_file = label_dir / f"{stem}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                
                src_img = img_dir / f"{stem}.jpg"
                if src_img.exists():
                    dst_img = image_link_dir / f"{stem}.jpg"
                    if not dst_img.exists():
                        os.symlink(src_img.resolve(), dst_img.resolve())
    
    yaml_content = f"""path: {output_dir}
train: images/train
val: images/test
nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"Converted to YOLO format at {output_dir}")
    print(f"- Train labels: {len(list((output_dir / 'labels' / 'train').glob('*.txt')))}")
    print(f"- Test labels: {len(list((output_dir / 'labels' / 'test').glob('*.txt')))}")

if __name__ == '__main__':
    main()