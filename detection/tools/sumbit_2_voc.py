import os

# Implement by fastvision

def prepare_folders(output_dir, year):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_dir = os.path.join(output_dir, 'results', f'VOC{year}', 'Main') # .../VOC2012/Main
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    return base_dir

def main(predicts, output_dir, prefix='comp3_det_test_', year=2012):
    '''
    :param predicts: dict : {'car' : [(2009_000026, 0.949297, 172.000000, 233.000000, 191.000000, 248.000000), ...], 'person' : [(), (), ()]}
                            img_id, score, xmin, ymin, xmax, ymax
                            the xmin of image is 1, not 0
    :return:
    '''

    base_dir = prepare_folders(output_dir=output_dir, year=year)

    for category_name, predictions in predicts.items():
        file_path = os.path.join(base_dir, f'{prefix}{category_name}.txt')
        f = open(file_path, 'w')
        for obj in predictions:
            f.write(f'{obj[0]} {obj[1]} {obj[2]} {obj[3]} {obj[4]} {obj[5]}\n') # img_id, score, xmin, ymin, xmax, ymax
        f.close()

# Example
main({'car' : [('2009_000026', 0.949297, 172.000000, 233.000000, 191.000000, 248.000000), ('2009_000026', 0.949297, 172.000000, 233.000000, 191.000000, 248.000000)], 'person' : [('2009_000026', 0.949297, 172.000000, 233.000000, 191.000000, 248.000000)]}, './', prefix='comp3_det_test_', year=2012)