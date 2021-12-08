import os
from glob import glob
import shutil
import xml.etree.ElementTree as ET
from multiprocessing import Pool
import tqdm

'''
input :
/home/VOCdevkit/VOC2012 :
    Annotations :
        *.xml
    ImageSets :
        Main :
            train.txt
            val.txt
    JPEGImages :
        *.jpg
    SegmentationClass (not necessary)
    SegmentationObject (not necessary)

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



def load_voc_annotation(annotation_path, category_names_idx_map):
    '''
    :param annotation_path: absolute annotation path
    :write: category_id, xmin, ymin, xmax, ymax
    '''
    root = ET.parse(annotation_path).getroot()

    labels = []
    for obj in root.findall('object'):
        category_name = obj.find('name').text.strip()
        category_id = category_names_idx_map[category_name]

        bndbox = obj.find('bndbox')
        x_min = int(float(bndbox.find('xmin').text.strip()))
        y_min = int(float(bndbox.find('ymin').text.strip()))
        x_max = int(float(bndbox.find('xmax').text.strip()))
        y_max = int(float(bndbox.find('ymax').text.strip()))

        labels.append((category_id, x_min, y_min, x_max, y_max))

    return labels

def process_voc_dataset(img_id, input_annotation_dir, input_image_dir, output_images_dir, output_labels_dir, category_names_idx_map):
    input_image_path = glob(os.path.join(input_image_dir, f'{img_id}.*'))
    assert len(input_image_path) == 1, f"please check {img_id} with {len(input_image_path)} match images : {input_image_path}"
    input_annotation_path = os.path.join(input_annotation_dir, f'{img_id}.xml')

    # copy image to dest image folder
    shutil.copy(input_image_path[0], output_images_dir)

    # process annotations and write to dest label folder
    root = ET.parse(input_annotation_path).getroot()

    f = open(os.path.join(output_labels_dir, f'{img_id}.txt'), 'w')

    for obj in root.findall('object'):
        category_name = obj.find('name').text.strip()
        category_id = category_names_idx_map[category_name]

        bndbox = obj.find('bndbox')
        x_min = int(float(bndbox.find('xmin').text.strip()))
        y_min = int(float(bndbox.find('ymin').text.strip()))
        x_max = int(float(bndbox.find('xmax').text.strip()))
        y_max = int(float(bndbox.find('ymax').text.strip()))

        f.write(f'{category_id} {x_min} {y_min} {x_max} {y_max}\n')

    f.close()

def checkdirs(annotation_dir, image_dir, imgset_path, output_images_dir, output_labels_dir):
    if not os.path.exists(annotation_dir):
        raise Exception(f'annotation dir {annotation_dir} not exists')

    if not os.path.exists(image_dir):
        raise Exception(f'image dir {image_dir} not exists')

    if not os.path.exists(imgset_path):
        raise Exception(f'imageSet path {imgset_path} not exists')

    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

def load_imagesSets(imgset_path):
    with open(imgset_path, 'r') as f:
        lines = f.readlines()

    imgsets = []
    for line in lines:
        imgsets.append(line.strip().split()[0].strip())
    return imgsets

def trans_voc_2_fastvision(voc_dir, imgset_name, output_dir, category_names_idx_map, works=1):
    '''
    :param voc_dir: /home/VOCdevkit/VOC2012
    :param imgset_name: train.txt
    :param output_dir: /home/dataset/voc2012/train
    :param category_list: ['aeroplane', 'bicycle', 'bird', ...]
    :return:
    '''
    input_annotation_dir = os.path.join(voc_dir, 'Annotations')
    input_image_dir = os.path.join(voc_dir, 'JPEGImages')
    input_imgset_path = os.path.join(voc_dir, 'ImageSets', 'Main', imgset_name)

    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')

    checkdirs(input_annotation_dir, input_image_dir, input_imgset_path, output_images_dir, output_labels_dir)

    imgsets = load_imagesSets(input_imgset_path)

    # ========================  tqdm configuration
    pbar = tqdm.tqdm(total=len(imgsets))
    pbar.set_description('Processing ')
    update = lambda *args: pbar.update()
    # ============================================

    pool = Pool(works)
    for img_id in imgsets:
        pool.apply_async(process_voc_dataset, args=(img_id, input_annotation_dir, input_image_dir, output_images_dir, output_labels_dir, category_names_idx_map, ), callback=update)
    pool.close()
    pool.join()


if __name__ == '__main__':
    category_list = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    category_names_idx_map = {name : idx for idx, name in enumerate(category_list)}

    trans_voc_2_fastvision(voc_dir=r'S:\VOCdevkit\VOC2012', imgset_name=r'train.txt', output_dir=r'S:\datasets\voc2012\train', category_names_idx_map=category_names_idx_map, works=8)
    trans_voc_2_fastvision(voc_dir=r'S:\VOCdevkit\VOC2012', imgset_name=r'val.txt', output_dir=r'S:\datasets\voc2012\val', category_names_idx_map=category_names_idx_map, works=8)


