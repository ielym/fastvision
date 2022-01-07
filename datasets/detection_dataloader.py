import yaml
import os
from glob import glob
import tqdm

import numpy as np
import cv2
import multiprocessing

import torch
from torch.utils.data import DataLoader, Dataset, distributed

from .common.augmentation import Augmentation, HorizontalFlip, VerticalFlip, Normalization
from .common.padding import Padding
from ..detection.tools import xyxy2xywhn, xyxy2xywh
from fastvision.detection.plot import draw_box_label


class BaseDataset(Dataset):
    def __init__(self, samples, input_size, max_det):

        self.samples = samples

        self.max_det = max_det

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
                                Normalization(means_stds=Augmentation.imagenetNorm(mode='rgb'), p=1.0),
                            ], mode='detect')

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
        img, label = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0)

def _load_samples(img_name, images_dir, labels_dir, samples):
    img_id = img_name.split('.')[0]

    image_path = os.path.join(images_dir, img_name)
    label_path = os.path.join(labels_dir, f'{img_id}.txt')

    with open(label_path, 'r') as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        category_id, xmin, ymin, xmax, ymax = line.strip().split()
        labels.append((float(category_id), float(xmin), float(ymin), float(xmax), float(ymax)))
    samples.append((image_path, labels))

def load_samples(data_dir, prefix, num_workers, cache, use_cache):

    if use_cache:
        with open(os.path.join(cache, f'{prefix}.txt'), 'r') as f:
            samples = eval(f.read())

        print(f'Use {prefix} data from cache {cache} {prefix}.txt')
        return samples

    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    img_names = os.listdir(images_dir)

    pool = multiprocessing.Pool(max(num_workers, 1))
    mgr = multiprocessing.Manager()
    samples = mgr.list()

    # ------------- tqdm with multiprocessing -------------
    pbar = tqdm.tqdm(total=len(img_names))
    pbar.set_description(f'Extract {prefix} dataset ')
    update_tqdm = lambda *args: pbar.update()
    # -----------------------------------------------------
    for img_name in img_names:
        pool.apply_async(_load_samples, args=(img_name, images_dir, labels_dir, samples, ), callback=update_tqdm)

    pool.close()
    pool.join()
    pbar.close()

    if cache:
        with open(os.path.join(cache, f'{prefix}.txt'), 'w') as f:
            f.write(str(samples))
        print(f'Save {prefix} data to cache {cache} {prefix}.txt')

    return samples

def create_dataloader(prefix, data_dir, batch_size, input_size, device, num_workers=0, cache='./cache', use_cache=False, shuffle=True, pin_memory=True, drop_last=False, max_det=200):

    samples = load_samples(data_dir, prefix, num_workers, cache, use_cache)

    dataset = BaseDataset(samples, input_size, max_det)

    loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                # sampler=distributed.DistributedSampler(datasets, shuffle=shuffle),
                drop_last=drop_last,
                num_workers=num_workers if device.type != 'cpu' else 0,
                collate_fn=BaseDataset.collate_fn,
            )

    return loader

def show_dataset(prefix, data_dir, category_names, num_workers, cache, use_cache):
    samples = load_samples(data_dir, prefix, num_workers, cache, use_cache)

    for sample in samples:
        img_path = sample[0]
        img = cv2.imread(img_path)

        labels = sample[1]

        for label in labels:
            category_idx, xmin, ymin, xmax, ymax = label
            draw_box_label(img, (int(xmin), int(ymin), int(xmax), int(ymax)), text=category_names[int(category_idx)], line_color=int(category_idx))

        cv2.imshow('img', img)
        cv2.waitKey(0)




