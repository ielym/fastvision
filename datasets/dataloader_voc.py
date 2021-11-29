import yaml
import os
from glob import glob

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
    def __init__(self, sample_ids, input_size, num_classes, max_det, category_names, jpeg_dir, annotaion_dir):

        self.image_dir = jpeg_dir
        self.annotaion_dir = annotaion_dir

        self.sample_ids = sample_ids

        self.max_det = max_det
        self.num_classes = num_classes
        self.category_names = category_names

        self.input_size = input_size
        if isinstance(input_size, int):
            self.input_height = input_size
            self.input_width = input_size
        else:
            self.input_height = input_size[0]
            self.input_width = input_size[1]

        self.category_names_idx_map = {}
        for idx, k in enumerate(category_names):
            self.category_names_idx_map[k] = idx

        self.augmentation = Augmentation([
                                HorizontalFlip(p=0.5),
                                VerticalFlip(p=0.1),
                                Normalization(Augmentation.imagenetNorm(mode='rgb'), p=1.0),
                            ])

    def __len__(self):
        return len(self.sample_ids)

    def load_label(self, annotation_path):

        root = ET.parse(annotation_path).getroot()

        labels = []
        for obj in root.findall('object'):
            category_name = obj.find('name').text.strip()
            category_id = self.category_names_idx_map[category_name]

            bndbox = obj.find('bndbox')
            x_min = int(float(bndbox.find('xmin').text.strip()))
            y_min = int(float(bndbox.find('ymin').text.strip()))
            x_max = int(float(bndbox.find('xmax').text.strip()))
            y_max = int(float(bndbox.find('ymax').text.strip()))

            labels.append([category_id, x_min, y_min, x_max, y_max])
        labels = np.array(labels, dtype=np.float32).reshape([-1, 5])
        return labels

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
        img_id = self.sample_ids[idx, 0]

        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        annotation_path = os.path.join(self.annotaion_dir, f'{img_id}.xml')

        # ======================================== process image ========================================
        resized_img, ori_hw, resized_hw, resized_ratio_hw = self.load_image(img_path, mode='rgb')
        img, padding_position = Padding(resized_img, input_size=(self.input_height, self.input_width), color=(114, 114, 114), align='center')

        # ======================================== process label ========================================
        ori_label = self.load_label(annotation_path)
        label = self.preprocess_labels(ori_label, resized_ratio_hw, padding_position)


        if len(label):
            label[:, 1:] = xyxy2xywhn(label[:, 1:], heigth=self.input_height, width=self.input_width)
            # label[:, 1:] = xyxy2xywh(label[:, 1:])
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

def create_dataloader(data_dir, imgset_name, category_names, batch_size, num_workers=0, shuffle=True, pin_memory=True, drop_last=False, input_size=640, num_classes=80, max_det=200):

    if num_workers > 0 and num_workers < 1:
        num_workers = int(multiprocessing.cpu_count() * num_workers)

    imageSets_dir = os.path.join(data_dir, 'ImageSets', 'Main')
    JPEGImages_dir = os.path.join(data_dir, 'JPEGImages')
    annotations_dir = os.path.join(data_dir, 'Annotations')

    img_ids = load_imagesSets(imageSets_dir, imgset_name)
    img_ids = np.array(img_ids).reshape([-1, 1])

    dataset = BaseDataset(img_ids, input_size, num_classes, max_det, category_names, JPEGImages_dir, annotations_dir)

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



