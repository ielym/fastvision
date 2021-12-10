import math
from torch.optim import lr_scheduler

def CosineLR(optimizer, steps, initial_lr, last_lr):
    '''
    :param steps: epochs or total_steps(epochs * batches)
    :param initial_lr:
    :param last_lr:
    :return:

    from initial_lr to last_lr by a single cosine curve
    initial_lr can be less than last_lr, also can be greater than last_lr
    '''

    lr_func = lambda cur_step: ((1 - math.cos(cur_step * math.pi / steps)) / 2) * (last_lr - initial_lr) + initial_lr
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    return scheduler

def LinearLR(optimizer, steps, initial_lr, last_lr):
    '''
    :param steps: epochs or total_steps(epochs * batches)
    :param initial_lr:
    :param last_lr:
    :return:

    from initial_lr to last_lr by a single line curve
    initial_lr can be less than last_lr, also can be greater than last_lr
    '''

    lr_func = lambda cur_step: (1 - cur_step / (steps - 1)) * (initial_lr - last_lr) + last_lr
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler

# epochs = 300
# init_lr = 1e-3
# minimum_lr = 1e-6
#
# lr1 = CosineLR(epochs, init_lr, minimum_lr)
# lr2 = LinearLR(epochs, init_lr, minimum_lr)
#
# lrs = []
# for i in range(epochs):
#     lrs.append(lr1(i))
#
# from matplotlib import pyplot as plt
# import numpy as np
#
# lrs = np.array(lrs)
# epoches = np.arange(len(lrs))
#
# plt.plot(epoches, lrs)
# plt.show()
#
# print(lrs)