import os
from glob import glob
import shutil
import xml.etree.ElementTree as ET
import multiprocessing
import tqdm
import json

'''
input :
/home/coco2017 :
    annotations:
        instances_train2017.json
        instances_val2017.json
        ...
    train2017 :
        *.jpg
    val2017 :
        *.jpg

============================================
output :
output_dir_name :
    train :
        images img_id.jpg
        labels img_id.txt
    val :
        images img_id.jpg
        labels img_id.txt
    test :
        images img_id.jpg
        labels img_id.txt
'''

def checkdirs(input_annotation_path, input_image_dir, output_images_dir, output_labels_dir):
    if not os.path.exists(input_annotation_path):
        raise Exception(f'annotation path {input_annotation_path} not exists')

    if not os.path.exists(input_image_dir):
        raise Exception(f'image dir {input_image_dir} not exists')


    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)


def trans_coco_2_fastvision(coco_dir, img_dir, annotation_name, output_dir, category_names_idx_map, works=1):
    '''
    :param coco_dir: /home/coco2017
    :param annotation_name: instances_train2017.json
    :param output_dir: /home/dataset/coco2017/train
    :param category_list: ['aeroplane', 'bicycle', 'bird', ...]
    :return:
    '''
    input_annotation_path = os.path.join(coco_dir, 'annotations', annotation_name)
    input_image_dir = os.path.join(coco_dir, img_dir)

    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')

    # checkdirs(input_annotation_path, input_image_dir, output_images_dir, output_labels_dir)

    data_dict = json.load(open(input_annotation_path, 'r')) # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

    coco_90_80_map = {}
    for category in data_dict['categories']:
        name = category['name']
        id_90 = category['id']
        coco_90_80_map[id_90] = category_names_idx_map[name]

    img_id_name_map = {}
    for img in data_dict['images']:
        img_id_name_map[img['id']] = img['file_name']


    records = {}
    for obj in tqdm.tqdm(data_dict['annotations'], desc='Extract '):
        img_id = obj['image_id']
        img_name = img_id_name_map[img_id]

        category_id_90 = obj['category_id']  # 90 categories
        category_id_80 = coco_90_80_map[category_id_90]

        x, y, w, h = obj['bbox']
        x_min = x
        y_min = y
        x_max = x + w
        y_max = y + h

        if not img_name in records.keys():
            records[img_name] = [(category_id_80, x_min, y_min, x_max, y_max)]
        else:
            records[img_name].append((category_id_80, x_min, y_min, x_max, y_max))

    for img_name, labels in tqdm.tqdm(records.items(), desc='Write '):
        input_image_path = os.path.join(input_image_dir, img_name)
        shutil.copy(input_image_path, output_images_dir)

        img_id = img_name.split('.')[0]
        with open(os.path.join(output_labels_dir, f'{img_id}.txt'), 'w') as f:
            for line in labels:
                category_id, x_min, y_min, x_max, y_max = line
                f.write(f'{category_id} {x_min} {y_min} {x_max} {y_max}\n')


if __name__ == '__main__':
    category_list = [
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                        'hair drier', 'toothbrush'
            ]

    category_names_idx_map = {name : idx for idx, name in enumerate(category_list)}

    trans_coco_2_fastvision(coco_dir=r'S:\coco2017', img_dir=r'train2017', annotation_name='instances_train2017.json', output_dir=r'S:\datasets\coco2017\train', category_names_idx_map=category_names_idx_map, works=8)
    trans_coco_2_fastvision(coco_dir=r'S:\coco2017', img_dir=r'val2017', annotation_name='instances_val2017.json', output_dir=r'S:\datasets\coco2017\val', category_names_idx_map=category_names_idx_map, works=8)


