import math
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right

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

def ExponentialLR(optimizer, steps, initial_lr, last_lr):
    p = (last_lr / initial_lr) ** (1 / steps)

    lr_func = lambda cur_step: initial_lr * p ** cur_step
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, milestones, min_ratio=0., cycle_decay=1., warmup_iters=1000, warmup_factor=1./10, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}".format(milestones)
            )
        self.milestones = [warmup_iters]+milestones
        self.min_ratio = min_ratio
        self.cycle_decay = cycle_decay
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha

            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        else:
            # which cyle is it
            cycle = min(bisect_right(self.milestones, self.last_epoch), len(self.milestones)-1)
            # calculate the fraction in the cycle
            fraction = min((self.last_epoch - self.milestones[cycle-1]) / (self.milestones[cycle]-self.milestones[cycle-1]), 1.)

            return [base_lr*self.min_ratio + (base_lr * self.cycle_decay**(cycle-1) - base_lr*self.min_ratio) *
                    (1 + math.cos(math.pi * fraction)) / 2
                    for base_lr in self.base_lrs]

# epochs = 100
# init_lr = 1e-10
# last_lr = 10
#
# lr1 = ExponentialLR(epochs, init_lr, last_lr)
# lr2 = LinearLR(epochs, init_lr, last_lr)
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