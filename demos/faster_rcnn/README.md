# 训练

+ 修改 `args.training = True`
+ Backbone 为 `fastvision` 中的模型及预训练权重
+ 数据为 `fastvision` 的格式 （类别索引从 0 开始）
+ 基本超参为 ：`SGD : bs = 1, lr = 1e-3, epoch=10, lr_step = 8/0.1` ，`Adam : lr *= 0.1`

+ 更换 Backbone :
  + 修改 `backbone_stride = 实际下采样倍数` ：
    + `vgg = 16` ，移除最后一个最大池化。
  + 修改 `backbone_output_channels = 实际输出通道数`：
    +  `vgg = 512`

```bash
python run.py
```



# 测试

+ 修改 `args.training = False`

```bash
python run.py
```



# Results

| Backbone | Train             | GPUs     | batch size | lr   | lr_decay | max_epoch | time/epoch | Test         | mAP50  |
| -------- | ----------------- | -------- | ---------- | ---- | -------- | --------- | ---------- | ------------ | ------ |
| VGG16    | VOC2012 Train     | 1 / V100 | 32         | 1e-2 | 8        | 10        | 5 min      | VOC2012 Val  | 0.4139 |
| VGG16    | VOC2012 Train_Val | 1 / V100 | 32         | 1e-2 | 8        | 10        | 10 min     | VOC2012 Val  | 0.6295 |
| VGG16    | VOC2012 Train_Val | 1 / V100 | 24         | 1e-2 | 8        | 20        | 10 min     | VOC2012 Test | 0.5673 |
| Res50    | VOC2012 Train     | 1 / V100 | 32         | 1e-2 | 8        | 10        |            | VOC2012 Val  |        |
| Res50    | VOC2012 Train_Val | 1 / V100 | 32         | 1e-2 | 8        | 10        |            | VOC2012 Test |        |
