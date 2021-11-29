import cv2

def Padding(ori_img, input_size, color=(114, 114, 114), align='center'):

    ori_height, ori_width = ori_img.shape[:2]
    target_height, target_widht = input_size[0], input_size[1]

    padding_height_double = target_height - ori_height
    padding_width_double = target_widht - ori_width

    if align == 'center':
        padding_height_half = padding_height_double / 2
        padding_width_half = padding_width_double / 2
        top, bottom = int(round(padding_height_half - 0.1)), int(round(padding_height_half + 0.1))
        left, right = int(round(padding_width_half - 0.1)), int(round(padding_width_half + 0.1))
    else:
        top = 0
        bottom = padding_height_double
        left = 0
        right = padding_width_double

    padding_img = cv2.copyMakeBorder(ori_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padding_img, (top, left, bottom, right)