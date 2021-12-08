import torch
import numpy as np
from fastvision.detection.tools import cal_iou_batch


class CalculateMAP:
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
        correct = np.zeros([y_pred.size(0), 2 + len(self.map_iou_values) ], dtype=np.float)

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
        iou_between_target_and_predict = cal_iou_batch(target_xyxy, predict_xyxy, mode='xyxy') # torch.Size([N, M])
        iou_mask = (iou_between_target_and_predict > self.map_iou_values[0]).float()

        # Match by Class
        cls_mask = (target_cls[:, None] == predict_cls).float() # torch.Size([N, M])

        # Match by both IOU and Class. If IOU(i, j) == True and Class(i, j) == True, then Matched_mask(i, j) == True
        matched_mask = ((iou_mask == 1) & (cls_mask == 1)).float() # torch.Size([N, M])
        matched_target_idx, matched_predict_idx = torch.where(matched_mask == 1)
        matched_target_predict_idx = torch.cat([matched_target_idx.view(-1, 1), matched_predict_idx.view(-1, 1)], dim=1).view(-1, 2)
        matched_target_predict_iou = iou_between_target_and_predict[matched_target_idx, matched_predict_idx].view(-1, 1)
        matched_target_predict_cls = target_cls[matched_target_idx].view(-1, 1)
        matched_target_predict_conf = predict_conf[matched_predict_idx].view(-1, 1)

        # For matched predicts-targets pairs, remove redundant pairs
        matched_matrix = torch.cat([matched_target_predict_idx, matched_target_predict_iou, matched_target_predict_cls, matched_target_predict_conf], dim=1)
        matched_matrix = matched_matrix.detach().cpu().numpy()
        matched_matrix = matched_matrix[np.argsort(-matched_matrix[:, 2]), ...] # sort by iou, from large to small
        matched_matrix = matched_matrix[np.unique(matched_matrix[:, 1], return_index=True)[1], ...] # remove redundant predicts, to make sure one predict just correspoind to one target
        matched_matrix = matched_matrix[np.unique(matched_matrix[:, 0], return_index=True)[1], ...] # remove redundant targets, to make sure one target just correspoind to one predict

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
        else: # voc2007, 11 points [0.0:1.1:0.1]
            ap = 0.0
            raise Exception('Not complete')

        return ap

    def _ap_per_class(self, total_positive, correct):

        ap_under_each_iou = np.zeros((len(self.map_iou_values), ), dtype=np.float)

        TP_cumsum = np.cumsum(correct, axis=0)
        FN_cumsum = total_positive - TP_cumsum
        FP_cumsum = np.cumsum(1 - correct, axis=0)

        recall_cumsum = TP_cumsum / (TP_cumsum + FN_cumsum + 1e-16)
        precision_cumsum = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-16)

        for iou_idx in range(correct.shape[1]):
            ap_under_each_iou[iou_idx] = self.compute_ap(recall_cumsum[:, iou_idx], precision_cumsum[:, iou_idx])

        return ap_under_each_iou

    def fetch(self):

        correct_all_images = np.concatenate(self.correct_all_images, axis=0)

        seen_all_target_cls = np.concatenate(self.seen_all_targets_cls, axis=0)
        seen_unique_target_cls = np.unique(seen_all_target_cls).tolist()

        ap_each_class_each_iou = np.zeros((len(seen_unique_target_cls), len(self.map_iou_values)), dtype=np.float)
        for category_idx in seen_unique_target_cls:

            cur_correct = correct_all_images[correct_all_images[:, 1] == category_idx, ...]
            cur_correct = cur_correct[np.argsort(-cur_correct[:, 0]), ...]  # sort by conf, from large to small

            total_positive = np.sum(seen_all_target_cls == category_idx)

            ap_cur_class_each_iou = self._ap_per_class(total_positive, cur_correct[:, 2:])
            ap_each_class_each_iou[seen_unique_target_cls.index(category_idx)] = ap_cur_class_each_iou

        map_each_iou = np.mean(ap_each_class_each_iou, axis=0)
        map_each_cls = np.mean(ap_each_class_each_iou, axis=1)

        return map_each_iou, map_each_cls, [int(cls_idx) for cls_idx in seen_unique_target_cls]