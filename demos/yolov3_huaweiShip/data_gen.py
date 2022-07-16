# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import math
import codecs
import random
import numpy as np
from glob import glob
import cv2
from PIL import Image
# from albumentations import (
#     HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Resize,
#     Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
#     IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop, PadIfNeeded,
#     IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize, RandomCrop
# )
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

from utils.box import xyxy2xywhn, xyxy2xywh, xywh2xyxy

from albumentations import MedianBlur, ChannelShuffle, Blur, GaussianBlur, OneOf

class DataAugmentation():

    def train_transforms(self, img_size):
        return Compose([
            OneOf([Blur(blur_limit=3, p=1), MedianBlur(blur_limit=3, p=1), GaussianBlur(blur_limit=(3, 3), p=1)], p=0.5),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ChannelShuffle(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)

    def val_transforms(self, img_size):
        return Compose([
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def ResizeByMax(image, labels, max_size):
    '''
    :param image: np.ndarray [H, W, c] RGB
    :param labels: np.ndarray [n, 4] xyxy, without normalization
    :param max_size:
    :return:
    '''
    ori_height, ori_width = image.shape[:2]

    # process image
    resize_ratio = max_size / max(ori_height, ori_width)
    resize_height = int(ori_height * resize_ratio)
    resize_width = int(ori_width * resize_ratio)
    image = cv2.resize(image, (resize_width, resize_height))

    # process label
    labels = labels * resize_ratio

    return image, labels

def Padding(image, label, size, fill_value=128):
    '''
    :param image: np.ndarray [H, W, c] RGB
    :param label: np.ndarray [n, 4] xyxy, without normalization
    :param size: int or tuple or list. if int size=(size, size) : (h, w)
    :return:
    '''
    if isinstance(size, int):
        size = (size, size)

    ori_height, ori_width, channel = image.shape

    if size[0] < ori_height:
        raise Exception("Padding's target height can not less than image's height")

    if size[1] < ori_width:
        raise Exception("Padding's target width can not less than image's width")

    padding_top = int((size[0] - ori_height) // 2)
    padding_left = int((size[1] - ori_width) // 2)

    padding_image = np.zeros((size[0], size[1], channel), dtype=image.dtype)
    padding_image.fill(fill_value)
    padding_image[padding_top:padding_top+ori_height, padding_left:padding_left+ori_width, :] = image

    label[:, [1, 3]] = label[:, [1, 3]] + padding_top
    label[:, [0, 2]] = label[:, [0, 2]] + padding_left

    return padding_image, label

def HorizontalFlip(image, label):
    '''
    :param image: np.ndarray [H, W, c] RGB
    :param label: np.ndarray [n, 4] xyxy, without normalization
    :return:
    '''
    image = cv2.flip(image, 1)
    ori_height, ori_width, _ = image.shape

    label_xywh = xyxy2xywh(label)
    label_xywh[:, 0] = ori_width - label_xywh[:, 0]
    label = xywh2xyxy(label_xywh)

    return image, label

def VerticalFlip(image, label):
    '''
    :param image: np.ndarray [H, W, c] RGB
    :param label: np.ndarray [n, 4] xyxy, without normalization
    :return:
    '''
    image = cv2.flip(image, 0)
    ori_height, ori_width, _ = image.shape

    label_xywh = xyxy2xywh(label)
    label_xywh[:, 1] = ori_height - label_xywh[:, 1]
    label = xywh2xyxy(label_xywh)

    return image, label

def HueSaturationValue(image, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation

    dtype = image.dtype  # uint8

    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains

    hue, saturate, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    hsv_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(saturate, lut_sat), cv2.LUT(val, lut_val)))
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    return image

def HistEqualize(image, adaptive=True):
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # adaptive : 在局部进行自适应直方图均衡化
    if adaptive:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

def Jitter(image, labels, jitter):
    '''
    :param image: np.ndarray [H, W, c] RGB
    :param label: np.ndarray [n, 4] xyxy, without normalization
    :return:
    '''

    def get_random(a, b):
        return np.random.rand() * (b - a) + a

    ori_height, ori_width, _ = image.shape
    new_height = int(ori_height * get_random(1 - jitter, 1 + jitter))
    new_width = int(ori_width * get_random(1 - jitter, 1 + jitter))

    # process image
    image = cv2.resize(image, (new_width, new_height))

    # process label
    ratio_h = new_height / ori_height
    ratio_w = new_width / ori_width
    labels[:, [0, 2]] = labels[:, [0, 2]] * ratio_w
    labels[:, [1, 3]] = labels[:, [1, 3]] * ratio_h
    return image, labels

def Mosaic01(images_labels, input_size, fill_value=128):

    merge_image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    merge_image.fill(fill_value)
    merge_label_xyxy = []
    merge_label_category = []

    center_x = merge_image.shape[1] // 2
    center_y = merge_image.shape[0] // 2

    for idx, (image, label_xyxy, label_category) in enumerate(images_labels):
        image, label_xyxy = ResizeByMax(image, label_xyxy, input_size // 2)
        height, width, _ = image.shape

        merge_label_category.append(label_category)

        if idx == 0:
            merge_image[center_y-height:center_y, center_x-width:center_x, :] = image
            label_xyxy[:, [0, 2]] = label_xyxy[:, [0, 2]] + (center_x - width)
            label_xyxy[:, [1, 3]] = label_xyxy[:, [1, 3]] + (center_y - height)
            merge_label_xyxy.append(label_xyxy)
        elif idx == 1:
            merge_image[center_y-height:center_y, center_x:center_x+width, :] = image
            label_xyxy[:, [0, 2]] = label_xyxy[:, [0, 2]] + center_x
            label_xyxy[:, [1, 3]] = label_xyxy[:, [1, 3]] + (center_y - height)
            merge_label_xyxy.append(label_xyxy)
        elif idx == 2:
            merge_image[center_y:center_y+height, center_x-width:center_x, :] = image
            label_xyxy[:, [0, 2]] = label_xyxy[:, [0, 2]] + (center_x - width)
            label_xyxy[:, [1, 3]] = label_xyxy[:, [1, 3]] + center_y
            merge_label_xyxy.append(label_xyxy)
        elif idx == 3:
            merge_image[center_y:center_y+height, center_x:center_x+width, :] = image
            label_xyxy[:, [0, 2]] = label_xyxy[:, [0, 2]] + center_x
            label_xyxy[:, [1, 3]] = label_xyxy[:, [1, 3]] + center_y
            merge_label_xyxy.append(label_xyxy)

    merge_label_xyxy = np.concatenate(merge_label_xyxy, axis=0)
    merge_label_xyxy = np.clip(merge_label_xyxy, 0, input_size-1)

    merge_label_category = np.concatenate(merge_label_category, axis=0)

    return merge_image, merge_label_xyxy, merge_label_category.reshape(-1)

def mosaic4(images, labels, input_size):

    mosaic_border = [-input_size // 2, -input_size // 2]
    yc, xc = (int(random.uniform(-x, 2 * input_size + x)) for x in mosaic_border)  # mosaic center x, y

    for idx, (image, label) in enumerate(zip(images, labels)):
        h, w, c = image.shape

        if idx == 0:  # top left
            img4 = np.full((input_size * 2, input_size * 2, c), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif idx == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, input_size * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif idx == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(input_size * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif idx == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, input_size * 2), min(input_size * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

    cv2.imshow('masaic', img4)


class BaseDataset(Dataset):
    def __init__(self, samples, input_size, mode):
        self.samples = samples
        self.input_size = input_size
        self.mode = mode

        self.augmentation = DataAugmentation()
        self.train_transforms = self.augmentation.train_transforms(img_size=input_size)
        self.val_transforms = self.augmentation.val_transforms(img_size=input_size)

    def __len__(self):
        return len(self.samples)


    def load_image(self, img_path):
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = np.array(img)

        return img

    def load_label(self, label_path):

        with open(label_path, 'r') as f:
            lines = f.readlines()

        labels = []
        for line in lines:
            line = line.strip()
            category_idx, xmin, ymin, xmax, ymax = line.split()
            labels.append((category_idx, xmin, ymin, xmax, ymax))

        labels = np.array(labels, dtype=np.float32).reshape([-1, 5]) # [category_idx, xmin, ymin, xmax, ymax]

        return labels

    def preprocess_image_label(self, image_path, label_path):

        image = self.load_image(image_path)
        label = self.load_label(label_path)
        label_category = label[:, 0]
        label_xyxy = label[:, 1:]

        # mosaic4([image, image, image, image], [label, label, label, label], 640)

        # Jitter # 需要在ResizeByMax的前面
        if self.mode == 'train' and random.random() <= 0.5:
            image, label_xyxy = Jitter(image, label_xyxy, jitter=0.3)

        # Resize max side to input_size
        image, label_xyxy = ResizeByMax(image, label_xyxy, self.input_size)

        # HorizontalFlip
        if self.mode == 'train' and random.random() <= 0.5:
            image, label_xyxy = HorizontalFlip(image, label_xyxy)

        # VerticalFlip
        if self.mode == 'train' and random.random() <= 0.5:
            image, label_xyxy = VerticalFlip(image, label_xyxy)

        # HistEqualize
        if self.mode == 'train' and random.random() <= 0.5:
            image = HistEqualize(image)

        # HueSaturationValue
        if self.mode == 'train' and random.random() <= 0.5:
            image = HueSaturationValue(image, hgain=0.015, sgain=0.7, vgain=0.4)

        # image_show = image.copy()
        # for line in label_xyxy:
        #     xmin = line[0]
        #     ymin = line[1]
        #     xmax = line[2]
        #     ymax = line[3]
        #     image_show = cv2.rectangle(image_show, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=[0, 0, 255], thickness=2)
        # cv2.imshow('image_show', cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR))

        return image, label_xyxy, label_category

    def __getitem__(self, idx):
        img_path = self.samples[idx, 0]
        label_path = self.samples[idx, 1]
        image, label_xyxy, label_category = self.preprocess_image_label(img_path, label_path)

        if self.mode == 'train':
            all_images_labels = [(image, label_xyxy, label_category)]
            for _ in range(4 - 1):
                random_sample = random.choice(self.samples)
                image_random, label_xyxy_random, label_category_random = self.preprocess_image_label(random_sample[0], random_sample[1])
                all_images_labels.append((image_random, label_xyxy_random, label_category_random))

            image, label_xyxy, label_category = Mosaic01(all_images_labels, self.input_size, fill_value=128) # 中心点是新图像的中心，输出图像尺寸=input_size， image_size=input_size/2 ： anchor/1

            # show_image = image.copy()
            # for line in label_xyxy:
            #     xmin = int(line[0])
            #     ymin = int(line[1])
            #     xmax = int(line[2])
            #     ymax = int(line[3])
            #     show_image = cv2.rectangle(show_image, (xmin, ymin), (xmax, ymax), color=[255, 0, 0], thickness=2)
            # cv2.imshow('merge_image', cv2.cvtColor(show_image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
            image = self.train_transforms(image=image)['image'].to(dtype=torch.float32)
            image = image / 255.
        else:
            image, label_xyxy = Padding(image, label_xyxy, self.input_size, fill_value=128)
            image = self.val_transforms(image=image)['image'].to(dtype=torch.float32)
            image = image / 255.

        label_xyxy = xyxy2xywhn(label_xyxy, self.input_size, self.input_size)
        expand_labels = np.zeros([label_xyxy.shape[0], 6], dtype=np.float32) # batch_idx, cls_idx, xywh
        expand_labels[:, 1] = label_category
        expand_labels[:, 2:] = label_xyxy
        labels = torch.tensor(expand_labels, dtype=torch.float32)

        return image, labels

    @staticmethod
    def collate_fn(batch):
        img, label = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0)

def create_dataset(base_dir, input_size, mode):

    label_dir = os.path.join(base_dir, 'labels')
    image_dir = os.path.join(base_dir, 'images')

    img_names = os.listdir(image_dir)

    samples = []
    for img_name in img_names:
        base_name = img_name.split('.')[0]

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, f'{base_name}.txt')

        samples.append((img_path, label_path))
    samples = np.array(samples).reshape([-1, 2])
    np.random.shuffle(samples)

    print(f'total samples: {len(samples)}')

    dataset = BaseDataset(samples, input_size, mode)
    return dataset

if __name__ == '__main__':
    dataset = create_dataset(r'S:\datasets\huawei_det\V1\train', 608, 'train')

    for (img, label) in dataset:
        print(img.size(), label.size())
        # print(img.size())
        # img = img.numpy()
        # img = np.transpose(img, [1, 2, 0])
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #
        # for line in label:
        #     xywh = (line[2:] * 608).numpy()
        #
        #     xmin = xywh[0] - xywh[2] / 2
        #     ymin = xywh[1] - xywh[3] / 2
        #     xmax = xywh[0] + xywh[2] / 2
        #     ymax = xywh[1] + xywh[3] / 2
        #
        #     img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=[0, 0, 255], thickness=2)
        #
        # cv2.imshow('img', img)
        # cv2.waitKey()