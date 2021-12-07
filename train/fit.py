import torch
from tqdm import tqdm, trange
from fastvision.detection.tools import cal_iou_batch

import numpy as np

class Fit():

    def __init__(self, model, device, optimizer, loss, end_epoch, start_epoch=0, train_loader=None, val_loader=None, test_loader=None, data_dict=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss = loss
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = val_loader

        self.data_dict = data_dict


    def run_epoches(self):
        self._test()

        # for epoch in range(self.start_epoch, self.end_epoch):
            # self._train(epoch)

            # if self.val_loader:
            #     self._val()

            # if self.test_loader:
            #     self._test()


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

    def _val(self):
        self.model.eval()
        with torch.no_grad():
            with tqdm(self.train_loader) as t:
                for batch_idx, (images, labels) in enumerate(t):

                    if self.device.type == 'cuda':
                        images = images.cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)

                    pred = self.model(images)
                    loss = self.loss(pred, labels)

                    t.set_description(f"Validation")
                    t.set_postfix(batch=batch_idx + 1, loss=loss.item())

    def process_batch(self, target_cls, target_xyxy, predict_conf, predict_cls, predict_xyxy, iouv):
        for i in range(len(target_cls)):
            if target_cls[i] == 17:
                target_cls[i] = 57

        iou_between_target_and_predict = cal_iou_batch(target_xyxy, predict_xyxy, mode='xyxy')

        match_idx_x, match_idx_y = torch.where((iou_between_target_and_predict >= iouv[0]) & (target_cls.view(-1, 1) == predict_cls.view(1, -1)))  # IoU above threshold and classes match
        num_matches = len(match_idx_x)

        correct = torch.zeros([predict_cls.size(0), iouv.size(0)], dtype=torch.bool)

        if num_matches:
            matches_idx = torch.stack([match_idx_x, match_idx_y], 1)
            matches_iou = iou_between_target_and_predict[match_idx_x, match_idx_y].view(-1, 1)
            matches_idx_iou = torch.cat([matches_idx, matches_iou], 1)

            matches_idx_iou = matches_idx_iou[matches_idx_iou[:, 2].argsort(descending=True), ...]
            matches_idx_iou = matches_idx_iou[np.unique(matches_idx_iou[:, 1], return_index=True)[1], :]
            matches_idx_iou = matches_idx_iou[np.unique(matches_idx_iou[:, 0], return_index=True)[1], :]

            matches_idx_iou = torch.Tensor(matches_idx_iou).to(iouv.device)

            correct[matches_idx_iou[:, 1].long()] = matches_idx_iou[:, 2:3] >= iouv

        return correct

    def compute_ap(self, recall, precision, method='coco'):
        m_recall = np.concatenate(([0.0], recall, [1.0]))
        m_precision = np.concatenate(([1.0], precision, [0.0]))

        max_precision_after_each_recall = np.flip(np.maximum.accumulate(m_precision[::-1]))

        # methods: 'voc2007', 'voc2009', 'coco'
        if method == 'coco':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, m_recall, max_precision_after_each_recall), x)  # integrate
        elif method == 'voc2009':  # 'voc2009'
            i = np.where(m_recall[1:] != m_recall[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((m_recall[i + 1] - m_recall[i]) * max_precision_after_each_recall[i + 1])  # area under curve
        else: # voc2007, 11 points [0.0:1.1:0.1]
            ap = 0.0
            raise Exception('Not complete')

        return ap

    def ap_per_class(self, correct, pred_conf, pred_cls, target_cls):
        sorted_idx = np.argsort(-pred_conf)

        correct = correct[sorted_idx, ...]
        pred_conf = pred_conf[sorted_idx, ...]
        pred_cls = pred_cls[sorted_idx, ...]

        unique_classes = np.unique(target_cls)
        num_unique_class = unique_classes.shape[0]

        ap_under_each_iou = np.zeros((num_unique_class, correct.shape[1]))

        for cls_idx, cls in enumerate(unique_classes):
            cur_predict_map = (pred_cls == cls)
            cur_targets_map = (target_cls == cls)

            num_predict = np.sum(cur_predict_map)
            num_targets = np.sum(cur_targets_map)

            if num_predict == 0 or num_targets == 0:
                continue

            TP_cumsum = np.cumsum(correct[cur_predict_map==True, :], axis=0)
            FN_cumsum = num_targets - TP_cumsum
            FP_cumsum = np.cumsum(1 - correct[cur_predict_map==True, :], axis=0)

            recall_cumsum = TP_cumsum / (TP_cumsum + FN_cumsum + 1e-16)
            precision_cumsum = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-16)

            for iou_idx in range(correct.shape[1]):
                ap_under_each_iou[cls_idx, iou_idx] = self.compute_ap(recall_cumsum[:, iou_idx], precision_cumsum[:, iou_idx])

        return ap_under_each_iou

    def _test(self):
        from fastvision.detection.tools import non_max_suppression
        from fastvision.detection.tools import xywh2xyxy
        import numpy as np

        iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95

        names = {k: v for k, v in enumerate(self.data_dict['categories'])}

        jdict, stats, ap, ap_class = [], [], [], []

        for batch_idx, (images, labels) in enumerate(tqdm(self.val_loader)):

            if self.device.type == 'cuda':
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)


            head_out, results = self.model(images, val=True)

            input_whwh = torch.tensor(images.size()).to(images)[[3, 2, 3, 2]]
            labels[:, 2:] *= input_whwh

            for img_idx in range(images.size(0)):

                predict_conf, predict_cls, predict_xyxy = non_max_suppression(results[img_idx, ...], conf_thres=0.25, iou_thres=0.45, max_det=300)
                num_predict = len(predict_conf)

                target = labels[labels[:, 0] == img_idx, ...]
                num_target = len(target)

                if num_target and num_predict:
                    target_cls = target[:, 1]
                    target_xywh = target[:, 2:]
                    target_xyxy = xywh2xyxy(target_xywh)
                    correct = self.process_batch(target_cls.cpu(), target_xyxy.cpu(), predict_conf.detach().cpu(), predict_cls.detach().cpu(), predict_xyxy.detach().cpu(), iouv.cpu())
                else:
                    target_cls = torch.Tensor()
                    predict_conf = torch.Tensor()
                    predict_cls = torch.Tensor()
                    correct = torch.zeros(num_predict, iouv.size(0), dtype=torch.bool)

                stats.append((correct, predict_conf.detach(), predict_cls.detach(), target_cls))  # (correct, conf, pcls, tcls)

        all_correct = np.concatenate([x[0].cpu() for x in stats], axis=0)
        all_predict_conf = np.concatenate([x[1].cpu() for x in stats], axis=0)
        all_predict_cls = np.concatenate([x[2].cpu() for x in stats], axis=0)
        all_target_cls = np.concatenate([x[3].cpu() for x in stats], axis=0)

        ap_under_each_iou = self.ap_per_class(all_correct, all_predict_conf, all_predict_cls, all_target_cls) # [detected classes, 10] not all classes

        map_each_iou = np.mean(ap_under_each_iou, axis=0)
        map_each_cls = np.mean(ap_under_each_iou, axis=1)

        map_05_95 = np.mean(map_each_iou)

        print(map_each_iou)
        print(map_05_95)
