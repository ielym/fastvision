import numpy as np

class Augmentation():
    def __init__(self, augmentations):
        self.augmentations = augmentations

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

        has_label = True
        if isinstance(label, type(None)):
            has_label = False

        for func in self.augmentations:
            if has_label:
                img, label = func(img, label)
            else:
                img = func(img)
        if has_label:
            return img, label
        return img

import random
def HorizontalFlip(p=1.0):
    def call(img, label=None):
        '''
        :param img:
        :param label: [x_center, y_center, width, height]
        :return:
        '''
        has_label = True
        if isinstance(label, type(None)):
            has_label = False

        if random.random() <= p:
            img = np.fliplr(img)
            if has_label:
                base = 1 if (label[0, 2] < 1 and label[0, 3] <1) else img.shape[1]
                label[:, 0] = base - label[:, 0]
                return img, label
            return img
        else:
            if has_label:
                return img, label
            return img
    return call

def VerticalFlip(p=1.0):
    def call(img, label=None):
        '''
        :param img:
        :param label: [x_center, y_center, width, height]
        :return:
        '''
        has_label = True
        if isinstance(label, type(None)):
            has_label = False

        if random.random() <= p:
            img = np.flipud(img)
            if has_label:
                base = 1 if (label[0, 2] < 1 and label[0, 3] <1) else img.shape[1]
                label[:, 1] = base - label[:, 1]
                return img, label
            return img
        else:
            if has_label:
                return img, label
            return img
    return call

def Normalization(means_stds=None, p=1.0):
    if isinstance(means_stds, type(None)):
        means = np.array([0., 0., 0.], dtype=np.float32).reshape([1, 1, 3])
        stds = np.array([1., 1., 1.], dtype=np.float32).reshape([1, 1, 3])
        means_stds = (means, stds)

    def call(img, label=None):
        has_label = True
        if isinstance(label, type(None)):
            has_label = False
        if random.random() <= p:
            img = img / 255.
            img = (img - means_stds[0]) / means_stds[1]
        if has_label:
            return img, label
        return img
    return call