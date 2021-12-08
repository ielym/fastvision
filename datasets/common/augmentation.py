import numpy as np
import functools
from functools import wraps
import random

class Augmentation():
    def __init__(self, augmentations:list, mode='classification'):
        '''
        :param augmentations: list
        :param mode: classification detect
        '''
        self.augmentations = augmentations
        self.mode = mode

    @classmethod
    def imagenetNorm(cls, mode='rgb'):
        if mode == 'rgb':
            means = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape([1, 1, 3])
            stds = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape([1, 1, 3])
        else:
            means = np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape([1, 1, 3])
            stds = np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape([1, 1, 3])
        return (means, stds)

    def __call__(self, img, label=None):

        for func in self.augmentations:
            if self.mode == 'detect':
                img, label = func(img, label, mode=self.mode)
            else:
                img = func(img, mode=self.mode)

        if self.mode == 'detect':
            return img, label
        return img

class HorizontalFlip():

    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img, label=None, mode='classification'):
        '''
        :param img:
        :param label: [x_center, y_center, width, height]
        :return:
        '''
        execute = (random.random() <= self.p)

        if execute:
            img = np.fliplr(img)

        if execute and mode == 'detect':
            base = 1 if (label[0, 2] < 1 and label[0, 3] <1) else img.shape[1]
            label[:, 0] = base - label[:, 0]

        if mode == 'detect':
            return img, label
        return img

class VerticalFlip():

    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img, label=None, mode='classification'):
        '''
        :param img:
        :param label: [x_center, y_center, width, height]
        :return:
        '''
        execute = (random.random() <= self.p)

        if execute:
            img = np.flipud(img)

        if execute and mode == 'detect':
            base = 1 if (label[0, 2] < 1 and label[0, 3] <1) else img.shape[1]
            label[:, 1] = base - label[:, 1]

        if mode == 'detect':
            return img, label
        return img

class Normalization():

    def __init__(self, means_stds=None, p=1.0):

        self.p = p

        if isinstance(means_stds, type(None)):
            means = np.array([0., 0., 0.], dtype=np.float32).reshape([1, 1, 3])
            stds = np.array([1., 1., 1.], dtype=np.float32).reshape([1, 1, 3])
            self.means_stds = (means, stds)
        else:
            self.means_stds = means_stds

    def __call__(self, img, label=None, mode='classification'):

        execute = (random.random() <= self.p)

        if execute:
            img = img / 255.
            img = (img - self.means_stds[0]) / self.means_stds[1]

        if mode == 'detect':
            return img, label
        return img