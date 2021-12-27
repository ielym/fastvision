import numpy as np
import random
import cv2
import math

from fastvision.detection.tools import xywh2xyxy, xyxy2xywh

class Augmentation():
    def __init__(self, augmentations:list, mode='classification'):
        '''
        :param augmentations: list
        :param mode: classification detect
        '''
        self.augmentations = augmentations
        self.mode = mode
        self.lock = False

    def lock_prob(self):
        self.lock = True

    def unlock_prob(self):
        self.lock = False
        for func in self.augmentations:
            func.lock = False

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
                img, label, execute = func(img, label, mode=self.mode)
            else:
                img, execute = func(img, mode=self.mode)

            if self.lock and not func.lock:
                func.lock = True
                func._execute = execute

        if self.mode == 'detect':
            return img, label
        return img

class BGR2RGB():

    def __init__(self, p=1.0):

        self.p = p
        self.lock = False
        self._execute= False

    def __call__(self, img, label=None, mode='classification'):

        execute = (random.random() <= self.p) if not self.lock else self._execute

        if execute:
            img = img[:, :, ::-1]

        if mode == 'detect':
            return img, label, execute
        return img, execute

class Resize():

    def __init__(self, size, interpolation=cv2.INTER_LINEAR, resize_by='longer', p=1.0):
        '''
        :param size: int or a tuple:(target_height, target_width). if int, then resize longer edge to size:int, and the ${resize_by} edge scales according the the ratio.
        :param p:
        '''
        self.size = size
        self.interpolation = interpolation
        self.resize_by = resize_by
        self.p = p

        self.lock = False
        self._execute = False

    def __call__(self, img, label=None, mode='classification'):
        '''
        :param img:
        :param label: [x_center, y_center, width, height]
        :return:
        '''
        execute = (random.random() <= self.p) if not self.lock else self._execute

        if execute:

            ori_height, ori_width = img.shape[:2]

            if isinstance(self.size, int):
                ratio = self.size / max(ori_height, ori_width) if self.resize_by == 'longer' else self.size / min(ori_height, ori_width)
                ratio_height, ratio_width = ratio, ratio
            else:
                ratio_height = self.size[0] / ori_height
                ratio_width = self.size[1] / ori_width

            img = cv2.resize(img, (int(ori_width * ratio_width), int(ori_height * ratio_height)), interpolation=self.interpolation)

        if execute and mode == 'detect':
            label[:, 0] = label[:, 0] * ratio_width
            label[:, 1] = label[:, 1] * ratio_height
            label[:, 2] = label[:, 2] * ratio_width
            label[:, 3] = label[:, 3] * ratio_height

        if mode == 'detect':
            return img, label, execute
        return img, execute

class Padding():

    def __init__(self, size, align='center', color=(114, 114, 114), p=1.0):
        '''
        :param size: int or a tuple:(target_height, target_width). if int, then padding both edges to size.
        :param p:
        '''
        self.size = size
        self.align = align
        self.color = color
        self.p = p

        self.lock = False
        self._execute = False

    def _padding_center(self, img, padding_height_double, padding_width_double):
        padding_height_half = padding_height_double / 2
        padding_width_half = padding_width_double / 2
        top, bottom = int(round(padding_height_half - 0.1)), int(round(padding_height_half + 0.1))
        left, right = int(round(padding_width_half - 0.1)), int(round(padding_width_half + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        return img, top, left

    def _padding_lefttop(self, img, padding_height_double, padding_width_double):
        top = 0
        bottom = padding_height_double
        left = 0
        right = padding_width_double
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)
        return img, top, left

    def __call__(self, img, label=None, mode='classification'):
        '''
        :param img:
        :param label: [x_center, y_center, width, height]
        :return:
        '''
        execute = (random.random() <= self.p) if not self.lock else self._execute

        if execute:
            ori_height, ori_width = img.shape[:2]

            if isinstance(self.size, int):
                target_height, target_width = self.size, self.size
            else:
                target_height, target_width = self.size[0], self.size[1]

            if target_height < ori_height or target_width < ori_width:
                raise Exception("Padding target width and target height should no less than origin width and origin height")

            padding_height_double = target_height - ori_height
            padding_width_double = target_width - ori_width

            if self.align == 'center':
                img, top, left = self._padding_center(img, padding_height_double, padding_width_double)
            else:
                img, top, left =self._padding_lefttop(img, padding_height_double, padding_width_double)

        if execute and mode == 'detect':
            label[:, 0] = label[:, 0] + left
            label[:, 1] = label[:, 1] + top

        if mode == 'detect':
            return img, label, execute
        return img, execute

class CenterCrop():

    def __init__(self, size, p=1.0):
        self.size = size
        self.p = p

        self.lock = False
        self._execute = False

    def __call__(self, img, label=None, mode='classification'):
        '''
        :param img:
        :param size: int or a tuple:(target_height, target_width). If int, then crop (size, size) from center.
        :param label: [x_center, y_center, width, height]
        :return:
        '''
        execute = (random.random() <= self.p) if not self.lock else self._execute

        if execute:
            ori_height, ori_width = img.shape[:2]

            if isinstance(self.size, int):
                target_height, target_width = self.size, self.size
            else:
                target_height, target_width = self.size

            if target_height > ori_height or target_width > ori_width:
                raise Exception("Crop target width and target height should no bigger than origin width and origin height")

            xmin = (ori_width - target_width) // 2
            ymin = (ori_height - target_height) // 2
            xmax = min(xmin + target_width, ori_width) - 1
            ymax = min(ymin + target_height, ori_height) - 1

            img = img[ymin:ymax+1, xmin:xmax+1, :]

        if execute and mode == 'detect':
            xyxy = xywh2xyxy(label)

            xyxy[:, 0] = np.maximum(xyxy[:, 0], xmin) - xmin
            xyxy[:, 1] = np.maximum(xyxy[:, 1], ymin) - ymin
            xyxy[:, 2] = np.minimum(xyxy[:, 2], xmax) - xmin
            xyxy[:, 3] = np.minimum(xyxy[:, 3], ymax) - ymin

            choise_labels = []
            for box in xyxy:
                x1, y1, x2, y2 = box
                if (x2 - x1) * (y2 - y1) <= 0:
                    continue
                choise_labels.append((x1, y1, x2, y2))
            label = np.array(choise_labels).astype(label.dtype).reshape([-1, 4])
            label = xyxy2xywh(label)

        if mode == 'detect':
            return img, label, execute
        return img, execute

class RandomCrop():

    def __init__(self, size, p=1.0):
        self.size = size
        self.p = p

        self.lock = False
        self._execute = False

    def __call__(self, img, label=None, mode='classification'):
        '''
        :param img:
        :param size: int or a tuple:(target_height, target_width). If int, then crop (size, size) from center.
        :param label: [x_center, y_center, width, height]
        :return:
        '''
        execute = (random.random() <= self.p) if not self.lock else self._execute

        if execute:
            ori_height, ori_width = img.shape[:2]

            if isinstance(self.size, int):
                target_height, target_width = self.size, self.size
            else:
                target_height, target_width = self.size

            if target_height > ori_height or target_width > ori_width:
                raise Exception("Crop target width and target height should no bigger than origin width and origin height")


            xmin = max(np.random.randint(0, ori_width - target_width), 0)
            ymin = max(np.random.randint(0, ori_height - target_height), 0)
            xmax = min(xmin + target_width, ori_width) - 1
            ymax = min(ymin + target_height, ori_height) - 1

            img = img[ymin:ymax+1, xmin:xmax+1, :]

        if execute and mode == 'detect':
            xyxy = xywh2xyxy(label)

            xyxy[:, 0] = np.maximum(xyxy[:, 0], xmin) - xmin
            xyxy[:, 1] = np.maximum(xyxy[:, 1], ymin) - ymin
            xyxy[:, 2] = np.minimum(xyxy[:, 2], xmax) - xmin
            xyxy[:, 3] = np.minimum(xyxy[:, 3], ymax) - ymin

            choise_labels = []
            for box in xyxy:
                x1, y1, x2, y2 = box
                if (x2 - x1) * (y2 - y1) <= 0:
                    continue
                choise_labels.append((x1, y1, x2, y2))
            label = np.array(choise_labels).astype(label.dtype).reshape([-1, 4])
            label = xyxy2xywh(label)

        if mode == 'detect':
            return img, label, execute
        return img, execute

class HorizontalFlip():

    def __init__(self, p=1.0):
        self.p = p

        self.lock = False
        self._execute = False

    def __call__(self, img, label=None, mode='classification'):
        '''
        :param img:
        :param label: [x_center, y_center, width, height]
        :return:
        '''
        execute = (random.random() <= self.p) if not self.lock else self._execute
        if execute:
            img = np.fliplr(img)

        if execute and mode == 'detect':
            base = 1 if (label[0, 2] < 1 and label[0, 3] <1) else img.shape[1]
            label[:, 0] = base - label[:, 0]

        if mode == 'detect':
            return img, label, execute
        return img, execute

class VerticalFlip():

    def __init__(self, p=1.0):
        self.p = p

        self.lock = False
        self._execute = False

    def __call__(self, img, label=None, mode='classification'):
        '''
        :param img:
        :param label: [x_center, y_center, width, height]
        :return:
        '''
        execute = (random.random() <= self.p) if not self.lock else self._execute

        if execute:
            img = np.flipud(img)

        if execute and mode == 'detect':
            base = 1 if (label[0, 2] < 1 and label[0, 3] <1) else img.shape[0]
            label[:, 1] = base - label[:, 1]

        if mode == 'detect':
            return img, label, execute
        return img, execute

class Normalization():

    def __init__(self, means_stds=None, p=1.0):

        self.p = p

        if isinstance(means_stds, type(None)):
            means = np.array([0., 0., 0.], dtype=np.float32).reshape([1, 1, 3])
            stds = np.array([1., 1., 1.], dtype=np.float32).reshape([1, 1, 3])
            self.means_stds = (means, stds)
        else:
            self.means_stds = means_stds

        self.lock = False
        self._execute = False

    def __call__(self, img, label=None, mode='classification'):

        execute = (random.random() <= self.p) if not self.lock else self._execute

        if execute:
            img = img / 255.
            img = (img - self.means_stds[0]) / self.means_stds[1]

        if mode == 'detect':
            return img, label, execute
        return img, execute