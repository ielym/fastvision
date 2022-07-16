import torch
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from .iou import cal_iou_batch, cal_iou


class mean_average_precision:
    def __init__(self, map_iou_values):
        '''
        :param map_iou_values: list or np.ndarray : map_iou_values = np.linspace(0.5, 0.95, 10) to get [0.5  0.55 0.6  0.65 0.7  0.75 0.8  0.85 0.9  0.95]
        '''
        self.map_iou_values = map_iou_values

        self.correct_all_images = []
        self.seen_all_targets_cls = []

    def process_one(self, y_pred, y_true):
        '''
        :param y_pred: torch.size([M, 6]) [category, confidence, xmin, ymin, xmax, ymax] recommended ori image size
        :param y_true: torch.size([N, 5]) [category, xmin, ymin, xmax, ymax] recommended ori image size
        :return:
        '''
        '''
            seen_target_categories_counts : y_true == category_idx.size(0)
            TP + FP : y_pred.size(0)
            TP : how many y_pred are matched with y_true
            FN : total_positive - TP

            1. We should record all predict whether it's TP or FP, because we have to use their conf as PR curve's thresholds.
            2. We should record all predict categories for calculate AP for each class.
        '''

        # shape : [M, 2 + len(self.map_iou_values)] : [conf, predict_cls, 10 map threshold]
        correct = np.zeros([y_pred.size(0), 2 + len(self.map_iou_values)], dtype=np.float)

        predict_cls = y_pred[:, 0]
        predict_conf = y_pred[:, 1]
        predict_xyxy = y_pred[:, 2:]

        target_cls = y_true[:, 0]
        target_xyxy = y_true[:, 1:]

        if target_cls.size(0) != 0:
            self.seen_all_targets_cls.append(target_cls.detach().cpu().numpy())

        if y_pred.size(0) == 0:
            return

        # Match by IOU
        iou_between_target_and_predict = cal_iou_batch(target_xyxy, predict_xyxy, mode='xyxy')  # torch.Size([N, M])
        iou_mask = (iou_between_target_and_predict > self.map_iou_values[0]).float()

        # Match by Class
        cls_mask = (target_cls[:, None] == predict_cls).float()  # torch.Size([N, M])

        # Match by both IOU and Class. If IOU(i, j) == True and Class(i, j) == True, then Matched_mask(i, j) == True
        matched_mask = ((iou_mask == 1) & (cls_mask == 1)).float()  # torch.Size([N, M])

        # ================ for torch<1.0.1
        matched_target_idx, matched_predict_idx = np.where(matched_mask.detach().cpu().numpy() == 1)
        matched_target_idx = torch.from_numpy(matched_target_idx).to(matched_mask).long()
        matched_predict_idx = torch.from_numpy(matched_predict_idx).to(matched_mask).long()
        # ================ for torch>1.0.1
        # matched_target_idx, matched_predict_idx = torch.where(matched_mask == 1)
        # ===============================================================================================================
        matched_target_predict_idx = torch.cat([matched_target_idx.view(-1, 1), matched_predict_idx.view(-1, 1)], dim=1).view(-1, 2)
        matched_target_predict_iou = iou_between_target_and_predict[matched_target_idx, matched_predict_idx].view(-1, 1)
        matched_target_predict_cls = target_cls[matched_target_idx].view(-1, 1)
        matched_target_predict_conf = predict_conf[matched_predict_idx].view(-1, 1)

        # For matched predicts-targets pairs, remove redundant pairs
        matched_matrix = torch.cat(
            [matched_target_predict_idx.float(), matched_target_predict_iou, matched_target_predict_cls,
             matched_target_predict_conf], dim=1)
        matched_matrix = matched_matrix.detach().cpu().numpy()
        matched_matrix = matched_matrix[np.argsort(-matched_matrix[:, 2]), ...]  # sort by iou, from large to small
        matched_matrix = matched_matrix[np.unique(matched_matrix[:, 1], return_index=True)[
                                            1], ...]  # remove redundant predicts, to make sure one predict just correspoind to one target
        matched_matrix = matched_matrix[np.unique(matched_matrix[:, 0], return_index=True)[
                                            1], ...]  # remove redundant targets, to make sure one target just correspoind to one predict

        # Though a predict-target pair may match under self.map_iou_values[0], but for other map_iou threshold, it may not match.
        correct[:, 0] = predict_conf.detach().cpu().numpy()
        correct[:, 1] = predict_cls.detach().cpu().numpy()
        correct[matched_matrix[:, 1].astype(np.long), 2:] = matched_matrix[:, 2:3] > self.map_iou_values

        self.correct_all_images.append(correct)

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
        else:  # voc2007, 11 points [0.0:1.1:0.1]
            ap = 0.0
            raise Exception('Not complete')

        return ap

    def _ap_per_class(self, total_positive, correct):

        ap_under_each_iou = np.zeros((len(self.map_iou_values),), dtype=np.float)

        TP_cumsum = np.cumsum(correct, axis=0)
        FN_cumsum = total_positive - TP_cumsum
        FP_cumsum = np.cumsum(1 - correct, axis=0)


        recall_cumsum = TP_cumsum / (TP_cumsum + FN_cumsum + 1e-16)
        precision_cumsum = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-16)

        for iou_idx in range(correct.shape[1]):
            ap_under_each_iou[iou_idx] = self.compute_ap(recall_cumsum[:, iou_idx], precision_cumsum[:, iou_idx])

        return ap_under_each_iou

    def fetch(self):


        if len(self.correct_all_images) == 0:
            return np.zeros((len(self.map_iou_values),)), np.array([0]), [0]
        correct_all_images = np.concatenate(self.correct_all_images, axis=0)

        seen_all_target_cls = np.concatenate(self.seen_all_targets_cls, axis=0)
        seen_unique_target_cls = np.unique(seen_all_target_cls).tolist()

        ap_each_class_each_iou = np.zeros((len(seen_unique_target_cls), len(self.map_iou_values)), dtype=np.float)
        for category_idx in seen_unique_target_cls:

            cur_correct = correct_all_images[correct_all_images[:, 1] == category_idx, ...]
            cur_correct = cur_correct[np.argsort(-cur_correct[:, 0]), ...]  # sort by conf, from large to small

            if len(cur_correct) == 0:
                continue

            total_positive = np.sum(seen_all_target_cls == category_idx)

            ap_cur_class_each_iou = self._ap_per_class(total_positive, cur_correct[:, 2:])
            ap_each_class_each_iou[seen_unique_target_cls.index(category_idx)] = ap_cur_class_each_iou

        map_each_iou = np.mean(ap_each_class_each_iou, axis=0)
        map_each_cls = np.mean(ap_each_class_each_iou, axis=1)

        return map_each_iou, map_each_cls, [int(cls_idx) for cls_idx in seen_unique_target_cls]


class mean_average_precision_ultralytics:

    def __init__(self, iouv = torch.linspace(0.5, 0.95, 10)):

        self.stats = []
        self.iouv = iouv

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def plot_pr_curve(self, px, py, ap, save_dir='pr_curve.png', names=()):
        # Precision-recall curve
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        py = np.stack(py, axis=1)

        if 0 < len(names) < 21:  # display per-class legend if < 21 classes
            for i, y in enumerate(py.T):
                ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
        else:
            ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

        ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.savefig(Path(save_dir), dpi=250)
        plt.close()

    def plot_mc_curve(self, px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
        # Metric-confidence curve
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

        if 0 < len(names) < 21:  # display per-class legend if < 21 classes
            for i, y in enumerate(py):
                ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
        else:
            ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

        y = py.mean(0)
        ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.savefig(Path(save_dir), dpi=250)
        plt.close()

    def compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves
        # Arguments
            recall:    The recall curve (list)
            precision: The precision curve (list)
        # Returns
            Average precision, precision curve, recall curve
        """

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
        return ap, mpre, mrec

    def ap_per_class(self, tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:  True positives (nparray, nx1 or nx10).
            conf:  Objectness value from 0-1 (nparray).
            pred_cls:  Predicted object classes (nparray).
            target_cls:  True object classes (nparray).
            plot:  Plot precision-recall curve at mAP@0.5
            save_dir:  Plot save directory
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes = np.unique(target_cls)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        px, py = np.linspace(0, 1, 1000), []  # for plotting
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = (target_cls == c).sum()  # number of labels
            n_p = i.sum()  # number of predictions

            if n_p == 0 or n_l == 0:
                continue
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum(0)
                tpc = tp[i].cumsum(0)

                # Recall
                recall = tpc / (n_l + 1e-16)  # recall curve
                r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

                # Precision
                precision = tpc / (tpc + fpc)  # precision curve
                p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

                # AP from recall-precision curve
                for j in range(tp.shape[1]):
                    ap[ci, j], mpre, mrec = self.compute_ap(recall[:, j], precision[:, j])
                    if plot and j == 0:
                        py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

        # Compute F1 (harmonic mean of precision and recall)
        f1 = 2 * p * r / (p + r + 1e-16)
        names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
        names = {i: v for i, v in enumerate(names)}  # to dict

        if plot:
            self.plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
            self.plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
            self.plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
            self.plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

        i = f1.mean(0).argmax()  # max F1 index
        return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

    def process_one(self, detections, labels):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """

        iouv = self.iouv

        detections = detections.clone()

        if len(labels):
            correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
            iou = self.box_iou(labels[:, 1:], detections[:, :4])
            x = torch.where(
                (iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detection, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                matches = torch.Tensor(matches).to(iouv.device)
                correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv

            tcls = labels[:, 0].tolist()  # target class
            self.stats.append((correct.detach().cpu(), detections[:, 4].detach().cpu(), detections[:, 5].detach().cpu(), tcls))  # (correct, conf, pcls, tcls)

        else:
            tcls = []  # target class
            niou = iouv.numel()
            correct = torch.zeros(detections.shape[0], niou, dtype=torch.bool)
            self.stats.append((correct.cpu(), detections[:, 4].cpu(), detections[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)


        return correct

    def fetch(self):
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy

        names = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}

        p, r, ap, f1, ap_class = self.ap_per_class(*stats, plot=True, save_dir='../../../yolov3_lym/utils/', names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

        return mp, mr, map50, map



