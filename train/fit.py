import torch
import numpy as np

from tqdm import tqdm, trange

from fastvision.detection.tools import cal_iou_batch
from fastvision.detection.tools import non_max_suppression
from fastvision.detection.tools import xywh2xyxy
from fastvision.metrics import CalculateMAP
from fastvision.utils.checkpoints import SaveModel

class Fit():

    def __init__(self, model, device, optimizer, scheduler, loss, end_epoch, start_epoch=0, train_loader=None, val_loader=None, test_loader=None, data_dict=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss = loss
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler

        self.category_names = {k: v for k, v in enumerate(data_dict['categories'])}


    def run_epoches(self):

        for epoch in range(self.start_epoch, self.end_epoch):
            self._train(epoch)

            if self.val_loader:
                self._val()

            ckpt = {
                    'model': self.model,
                    'optimizer': self.optimizer.state_dict()
                }

            SaveModel(ckpt, 'last.pth', weights_only=True)

        if self.test_loader:
            self._test()

    def _train(self, epoch):
        assert self.train_loader, 'train_loader can not be None'

        self.model.train()
        with tqdm(self.train_loader) as t:
            for batch_idx, (images, labels) in enumerate(t):
                if self.device.type == 'cuda':
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                pred = self.model(images)

                self.optimizer.zero_grad()
                loss = self.loss(pred, labels)

                loss.backward()
                self.optimizer.step()

                t.set_description(f"Epoch {epoch + 1}")
                t.set_postfix(batch=batch_idx + 1, loss=loss.item())

                lr_each_param_groups = [x['lr'] for x in self.optimizer.param_groups]
                print(lr_each_param_groups)
                self.scheduler.step()
                print(self.optimizer.state_dict()['param_groups']) # dict_keys(['state', 'param_groups'])

    def _val(self):

        map_est = CalculateMAP(map_iou_values=np.linspace(0.5, 0.95, 10))

        self.model.eval()
        with torch.no_grad():
            with tqdm(self.train_loader) as t:
                for batch_idx, (images, labels) in enumerate(t):

                    if self.device.type == 'cuda':
                        images = images.cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)

                    head_out, results = self.model(images, val=True)

                    loss = self.loss(head_out, labels)

                    t.set_description(f"Validation")
                    t.set_postfix(batch=batch_idx + 1, loss=loss.item())

                    # calculate map
                    for img_idx in range(images.size(0)):
                        predict_conf, predict_cls, predict_xyxy = non_max_suppression(results[img_idx, ...], conf_thres=0.25, iou_thres=0.45, max_det=300)
                        predict = torch.cat([predict_cls.float(), predict_conf, predict_xyxy], dim=1)

                        target = labels[labels[:, 0] == img_idx, 1:]
                        target[:, 1:] = xywh2xyxy(target[:, 1:]) * torch.tensor(images.size()).to(target)[[3, 2, 3, 2]]

                        map_est.process_one(predict, target)

            map_each_iou, map_each_cls, map_each_cls_idx = map_est.fetch()

            print(f'loss : {loss.item()} map : {map_each_iou.tolist()}')


    def _test(self):

        pass


