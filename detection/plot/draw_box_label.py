import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from .get_color import get_color

plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus']=False

def draw_box_label(image, box, text='', line_width=2, line_color=(128, 128, 128), font_size=1, font_color=(255, 255, 255), bgr=True):
    assert isinstance(image, np.ndarray), f'Type of parameter image must be np.ndaary, not {type(image)}'

    # if isinstance(font_color, int):
    #     font_color = get_color(font_color, bgr=bgr)
    if isinstance(line_color, int):
        line_color = get_color(line_color, bgr=bgr)

    line_width = line_width or round(sum(image.shape[:2]) / 2 * 0.003)

    x_min, y_min, x_max, y_max = box
    p1, p2 = (int(x_min), int(y_min)), (int(x_max), int(y_max))

    image = cv2.rectangle(image, p1, p2, line_color, thickness=line_width, lineType=cv2.LINE_AA)

    if text:
        font_size = font_size or max(line_width - 1, 1)
        font_w, font_h = cv2.getTextSize(text, 0, fontScale=line_width / 3, thickness=font_size)[0]
        outside = int(y_min) - font_h - 3 >= 0
        p2 = (int(x_min) + font_w, int(y_min) - font_h - 3 if outside else p1[1] + font_h + 3)
        image = cv2.rectangle(image, p1, p2, line_color, -1, cv2.LINE_AA)
        image = cv2.putText(image, text, (p1[0], p1[1] - 2 if outside else p1[1] + font_h + 2), 0, line_width / 3, font_color, thickness=font_size, lineType=cv2.LINE_AA)

    return image

