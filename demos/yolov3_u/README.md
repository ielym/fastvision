# 1 网络结构

+ 该 YOLOV3 的网络结构实现与 U 版 YOLOV3 的网络结构一致。

+ 已将 U 版 YOLOV3 的 COCO 训练权重转成该版本的权重，可直接加载权重即可。

+ 转换后的权重地址 (`yolov3_weights_from_github_ultralytics.pth`) ：

  链接：https://pan.baidu.com/s/12vT4OUieZBwX1EmISo0Eew?pwd=4zoh 
  提取码：4zoh 

  

# 2 Loss

+ 该版本的 `utils/lossv3_u.py` 与 U 版的 loss 相同，但需要再检查。
+ 该版本的 `utils/lossv3.py` 与华为船只检测的 loss 类似，但有改动，且不确定哪些地方存在改动，需要检查。



# 3 数据增强

+  该版本的数据增强直接替换成为了华为船只检测的数据增强，未训练测试过。



# 4 训练策略

+ 该版本的训练策略与 U 版的不同
+ 建议使用华为船只检测中的 `_fit.py` 的训练策略，但是涉及到 `run.py` 等其他可能且未知位置的改动，需要再检查。



# 5 Other

+ 建议此版本仅作为 U版 的推理代码，训练建议使用华为船只检测的代码。
+ 推理运行 `inference.py` 
+ 数据格式为 fastvision 中的数据格式。