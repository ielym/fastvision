import torch
import numpy as np

from tqdm import tqdm, trange
from fastvision.utils.checkpoints import SaveModel

class Fit():

    def __init__(self, model, device, optimizer, scheduler, loss, metric, end_epoch, start_epoch=0, train_loader=None, val_loader=None, test_loader=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler

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

                metric = self.metric(pred, labels)

                self.optimizer.zero_grad()
                loss = self.loss(pred, labels)

                loss.backward()
                self.optimizer.step()

                t.set_description(f"Epoch {epoch + 1}")
                t.set_postfix(batch=batch_idx + 1, loss=loss.item(), metric=metric.item())

            self.scheduler.step()

    @torch.no_grad()
    def _val(self):

        self.model.eval()
        with tqdm(self.train_loader) as t:
            for batch_idx, (images, labels) in enumerate(t):

                if self.device.type == 'cuda':
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                pred = self.model(images)

                loss = self.loss(pred, labels)

                t.set_description(f"Validation")
                t.set_postfix(batch=batch_idx + 1, loss=loss.item())

    def _test(self):

        pass


