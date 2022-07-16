import numpy as np
from matplotlib import pyplot as plt
import re

# log_path = r'./logs/log.txt'
log_path = r'./logs/log_version2.txt'

with open(log_path, 'r') as f:
    lines = f.readlines()

loss_xy = []
loss_wh = []
loss_cls = []
loss_conf = []
record = True
for line in lines:
    line = line.strip()
    # if not line.startswith('e'):
    #     print(line)
    if r'90 / 90' in line:
        record = False
    if r'learning' in line:
        record = True

    compile = re.compile(r'(\d+\.\d+)\s(\d+\.\d+)\s(\d+\.\d+)\s(\d+\.\d+)')
    ret = compile.match(line)
    if ret and record:
        ret = ret.groups()
        loss_xy.append(float(ret[0]))
        loss_wh.append(float(ret[1]))
        loss_cls.append(float(ret[2]))
        loss_conf.append(float(ret[3]))

begin = 0
end = -1
loss_xy = loss_xy[begin:end]
loss_wh = loss_wh[begin:end]
loss_cls = loss_cls[begin:end]
loss_conf = loss_conf[begin:end]

xs = list(range(len(loss_xy)))
# plt.plot(xs, loss_xy, label='loss_xy', color='red')
# plt.plot(xs, loss_wh, label='loss_wh', color='orange')
# plt.plot(xs, loss_cls, label='loss_cls', color='blue')
plt.plot(xs, loss_conf, label='loss_conf', color='green')

plt.legend()
plt.savefig('./1e-4_loss_xy.png')
plt.show()