import yaml
import os
from glob import glob
import tqdm

import numpy as np
import cv2
import xml.etree.ElementTree as ET
import multiprocessing

import torch
from torch.utils.data import DataLoader, Dataset, distributed

from .common.augmentation import Augmentation, HorizontalFlip, VerticalFlip, Normalization
from .common.padding import Padding
from ..detection.tools import xyxy2xywhn, xyxy2xywh

class BaseDataset(Dataset):
    def __init__(self, samples, input_size, num_classes, max_det):

        self.samples = samples

        self.max_det = max_det
        self.num_classes = num_classes

        self.input_size = input_size
        if isinstance(input_size, int):
            self.input_height = input_size
            self.input_width = input_size
        else:
            self.input_height = input_size[0]
            self.input_width = input_size[1]

        self.augmentation = Augmentation([
                                HorizontalFlip(p=0.5),
                                VerticalFlip(p=0.1),
                                Normalization(Augmentation.imagenetNorm(mode='rgb'), p=1.0),
                            ])

    def __len__(self):
        return len(self.samples)

    def load_image(self, img_path, mode='rgb'):

        ori_img = cv2.imread(img_path)
        if mode == 'rgb':
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        ori_height, ori_width = ori_img.shape[:2]

        if isinstance(self.input_size, int):
            ratio = self.input_size / max(ori_height, ori_width)
            ratio_height, ratio_width = ratio, ratio
        else:
            ratio_height = self.input_height / ori_height
            ratio_width = self.input_width / ori_width

        resiezd_img = cv2.resize(ori_img, (int(ori_width * ratio_width), int(ori_height * ratio_height)), interpolation=cv2.INTER_LINEAR)
        resized_height, resized_width = resiezd_img.shape[:2]

        return resiezd_img, (ori_height, ori_width), (resized_height, resized_width), (ratio_height, ratio_width)

    def preprocess_labels(self, ori_label, resized_ratio_hw, padding_position):
        ori_label[:, 1] = ori_label[:, 1] * resized_ratio_hw[1] + padding_position[1]
        ori_label[:, 2] = ori_label[:, 2] * resized_ratio_hw[0] + padding_position[0]
        ori_label[:, 3] = ori_label[:, 3] * resized_ratio_hw[1] + padding_position[1]
        ori_label[:, 4] = ori_label[:, 4] * resized_ratio_hw[0] + padding_position[0]
        return ori_label

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path = sample[0]
        annotations = sample[1]

        # ======================================== process image ========================================
        img_path = os.path.join(img_path)
        resized_img, ori_hw, resized_hw, resized_ratio_hw = self.load_image(img_path, mode='rgb')
        img, padding_position = Padding(resized_img, input_size=(self.input_height, self.input_width), color=(114, 114, 114), align='center')

        # ======================================== process label ========================================
        ori_label = np.array(annotations, dtype=np.float32).reshape([-1, 5])
        label = self.preprocess_labels(ori_label, resized_ratio_hw, padding_position)


        if len(label):
            label[:, 1:] = xyxy2xywhn(label[:, 1:], heigth=self.input_height, width=self.input_width)
            img, label[:, 1:] = self.augmentation(img, label[:, 1:])

        labels_out = torch.zeros([len(label), 6], dtype=torch.float32)
        labels_out[:, 1:] = torch.from_numpy(label)

        img = img.transpose([2, 0, 1])
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img_out = torch.from_numpy(img)

        return img_out, labels_out

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0)

def load_imagesSets(path, imgset_name):
    imgset_pathes = glob(os.path.join(path, imgset_name))

    imgsets = []
    for imgset_path in imgset_pathes:
        with open(imgset_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            imgsets.append(line.strip().split()[0].strip())
    return imgsets

    # function which load VOC format annotation
def load_voc_annotation(annotation_path, img_path, category_names_idx_map, return_list):
    '''
    :param annotation_path: absolute annotation path
    :return: a list contains a single image's labels : [ (category_id, xmin, ymin, xmax, ymax), (category_id, xmin, ymin, xmax, ymax), ... ]
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

    return_list.append((img_path, labels))

def load_voc(data_dir, imgset_name, category_names, num_workers):
    '''
    :param data_dir:
    :param imgset_name: train.txt or val.txt or test.txt or 'car_*.txt'
    :param category_names: a list that contains category english name, or a dict contains {'a_category_name', a_category_int_id, ...}
    :return: type: list  -> [absolute_img_path, annotation_list : [ (category_id, xmin, ymin, xmax, ymax), (category_id, xmin, ymin, xmax, ymax), ... ] ]
    '''

    # if not specific (int) category index for category english name, then generate category_names_idx_map
    category_names_idx_map = category_names if isinstance(category_names, dict) else {name : idx for idx, name in enumerate(category_names)}

    # VOC standard folder path
    imageSets_dir = os.path.join(data_dir, 'ImageSets', 'Main')
    JPEGImages_dir = os.path.join(data_dir, 'JPEGImages')
    annotations_dir = os.path.join(data_dir, 'Annotations')

    # Load image ids from imageSets
    imgset_pathes = glob(os.path.join(imageSets_dir, imgset_name))
    img_ids = []
    for imgset_path in imgset_pathes:
        with open(imgset_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            img_ids.append(line.strip().split()[0].strip())

    # initial multiprocessing
    manager = multiprocessing.Manager()
    return_list = manager.list()
    pool = multiprocessing.Pool(max(1, num_workers))
    for img_id in img_ids:
        img_path = os.path.join(JPEGImages_dir, f'{img_id}.jpg')
        annotation_path = os.path.join(annotations_dir, f'{img_id}.xml')
        pool.apply_async(load_voc_annotation, args=(annotation_path, img_path, category_names_idx_map, return_list))
    pool.close()
    pool.join()

    return return_list

def create_dataloader(data_dir, imgset_name, category_names, batch_size, num_workers=0, shuffle=True, pin_memory=True, drop_last=False, input_size=640, num_classes=80, max_det=200, cache_dir='./cache_dir', use_cache=False):


    if num_workers > 0 and num_workers < 1:
        num_workers = min(int(multiprocessing.cpu_count() * num_workers), int(multiprocessing.cpu_count()))

    '''
    samples : a list:
                [
                    [absolute_img_path1, [(category_id, xmin, ymin, xmax, ymax), (category_id, xmin, ymin, xmax, ymax), ...]],
                    [absolute_img_path2, [(category_id, xmin, ymin, xmax, ymax), (category_id, xmin, ymin, xmax, ymax), ...]],
                    ...
                ]
    '''
    if use_cache:
        with open(os.path.join(cache_dir, imgset_name), 'r') as f:
            samples = eval(f.read())
    else:
        samples = load_voc(data_dir, imgset_name, category_names, num_workers)
        with open(os.path.join(cache_dir, imgset_name), 'w') as f:
            f.write(str(samples))


    dataset = BaseDataset(samples, input_size, num_classes, max_det)

    loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                # sampler=distributed.DistributedSampler(datasets, shuffle=shuffle),
                drop_last=drop_last,
                num_workers=num_workers,
                collate_fn=BaseDataset.collate_fn,
            )

    return loader



