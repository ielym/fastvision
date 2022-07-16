import torch
import torch.nn as nn
import torch.nn.functional as F
from .iou import wh_iou_batch, xywh_iou_batch
from .box import grid

class ComputeLoss(nn.Module):

    def __init__(self):
        super(ComputeLoss, self).__init__()

    def get_model(self, model):
        if hasattr(model, 'module'):
            return model.module
        else:
            return model

    def forward(self, predict_layers, target_all, model):
        '''
        :param predict_layers: list -> torch.Size([bs, 255, 7, 7]) torch.Size([bs, 255, 14, 14]) torch.Size([bs, 255, 28, 28])
        :param target_all: torch.tensor -> [bs_idx, cls_idx, x, y, w, h]
        :param model:
        :return:
        '''

        model = self.get_model(model)
        anchor_layers = model.anchors # list -> [torch.size([3, 2]), torch.size([3, 2]), torch.size([3, 2])] feature-scale corresponding to predict_layers

        loss_xy = torch.zeros(1).to(predict_layers[0])
        loss_wh = torch.zeros(1).to(predict_layers[0])
        loss_cls = torch.zeros(1).to(predict_layers[0])
        loss_conf = torch.zeros(1).to(predict_layers[0])

        for layer_idx in range(len(predict_layers)):

            # ==================================== anchors ====================================
            anchor = anchor_layers[layer_idx] # torch.Size([3, 2]) feature-scale
            num_anchors = anchor.size(0)

            # ==================================== predict ====================================
            bs, _, feature_height, feature_width = predict_layers[layer_idx].size()
            predict = predict_layers[layer_idx].permute(0, 2, 3, 1).view(bs, feature_height, feature_width, num_anchors, -1) # torch.Size([2, 14, 14, 3, 85]) feature-scale

            # ==================================== target ====================================
            target = target_all.clone() # torch.Size([10, 6]) normalization-scale [bs_idx, cls_idx, x, y, w, h]
            target[:, 2:] = target[:, 2:] * torch.tensor([feature_width, feature_height, feature_width, feature_height]).to(predict) # feature-scale

            # ==================================== target in which anchor ====================================
            # TODO : each target will be predicted in every layer, but it should be predicted by only a single layer
            # TODO : iou between anchor or predict bbox?
            iou_targets_and_anchors = wh_iou_batch(target[:, 4:], anchor) # torch.Size([10, 3])
            max_iou_value_targets_and_anchors, max_iou_idx_targets_and_anchors = torch.max(iou_targets_and_anchors, dim=1) # torch.Size([10]) torch.Size([10])
            anchors_corresponding_to_targets = anchor[max_iou_idx_targets_and_anchors, :] # torch.Size([10, 2])

            # ==================================== target in which cell ====================================
            target_grid_xy = torch.floor(target[:, 2:4])
            target_offset_xy = target[:, 2:4] - target_grid_xy

            # ==================================== summary target ====================================
            # bs_idx, cls_idx, x, y, w, h, grid_x, grid_y, offset_x, offset_y, matched_anchor_idx, matched_anchor_w, matched_anchor_h
            target = torch.cat([target, target_grid_xy, target_offset_xy, max_iou_idx_targets_and_anchors.unsqueeze(1), anchors_corresponding_to_targets], dim=1) # torch.Size([10, 14]) feature-scale

            # ==================================== summary predict ====================================+-
            predict_xy = torch.sigmoid(predict[..., 0:2]) # torch.Size([2, 14, 14, 3, 2]) offset
            # TODO : Shold clip predict_wh to feature size? Pay attention to the exp function which may cause gradient problem
            predict_wh = torch.exp(predict[..., 2:4]) * anchor.repeat(1, 1, 1, 1, 1) # torch.Size([2, 14, 14, 3, 2]) feature-scale
            grid_xy = grid(height=feature_height, width=feature_width, mode='xy').repeat(1, 1, 1, 1).unsqueeze(3).to(predict_xy) # torch.Size([1, 14, 14, 1, 2])
            predict_xywh = torch.cat([predict_xy+grid_xy, predict_wh], dim=4) # torch.Size([2, 14, 14, 3, 4]) feature-scale

            # ==================================== loss xy ====================================+-
            predict_xy_for_loss = predict[target[:, 0].long(), target[:, 7].long(), target[:, 6].long(), target[:, 10].long(), 0:2] # torch.Size([10, 2])
            target_xy_for_loss = target[:, 8:10] # torch.Size([10, 2])
            loss_xy += F.binary_cross_entropy_with_logits(predict_xy_for_loss, target_xy_for_loss)

            # ==================================== loss wh ====================================+-
            predict_wh_for_loss = predict[target[:, 0].long(), target[:, 7].long(), target[:, 6].long(), target[:, 10].long(), 2:4] # torch.Size([10, 2])
            target_wh_for_loss = torch.log((target[:, 4:6] / target[:, 11:13]) + 1e-14)
            loss_wh += F.mse_loss(predict_wh_for_loss, target_wh_for_loss)

            # ==================================== loss cls ====================================+-
            predect_cls_for_loss = predict[target[:, 0].long(), target[:, 7].long(), target[:, 6].long(), target[:, 10].long(), 5:] # torch.Size([10, 80])
            target_cls_for_loss = torch.zeros_like(predect_cls_for_loss)
            target_cls_for_loss[range(len(target_cls_for_loss)), target[:, 1].long()] = 1
            loss_cls += F.binary_cross_entropy_with_logits(predect_cls_for_loss, target_cls_for_loss)

            # ==================================== ignore mask ====================================+-
            # TODO : Can optimize the FOR LOOP ?
            mask_all_images = []
            for gt_idx in range(bs):
                predict_xywh_single_image = predict_xywh[gt_idx, ...] # torch.Size([14, 14, 3, 4])
                target_xywh_single_image = target[target[:, 0] == gt_idx][:, 2:6] # torch.Size([8, 4])

                iou_targets_and_predicts = xywh_iou_batch(predict_xywh_single_image.view(-1, 4), target_xywh_single_image) # torch.Size([588, 8])
                max_iou_value_targets_and_predicts, max_iou_idx_targets_and_predicts = torch.max(iou_targets_and_predicts, dim=1)  # torch.Size([588])

                mask_single_image = torch.zeros( (iou_targets_and_predicts.size(0), 1) ).to(predict) # torch.Size([588, 1])
                mask_single_image[max_iou_value_targets_and_predicts > 0.5] = -1 # ignore
                mask_single_image = mask_single_image.view(feature_height, feature_width, num_anchors, 1) # torch.Size([14, 14, 3, 1])
                mask_all_images.append(mask_single_image.unsqueeze(0))
            mask_all_images = torch.cat(mask_all_images, 0) # torch.Size([2, 14, 14, 3, 1])
            mask_all_images[target[:, 0].long(), target[:, 7].long(), target[:, 6].long(), target[:, 10].long(), ...] = 1 # positive

            # ==================================== loss conf ====================================+-
            predict_conf_for_loss = predict[..., 4:5][mask_all_images != -1] # torch.Size([1176])
            target_conf_for_loss = mask_all_images[mask_all_images != -1]
            loss_conf += F.binary_cross_entropy_with_logits(predict_conf_for_loss, target_conf_for_loss)

        print(loss_xy.item(), loss_wh.item(), loss_cls.item(), loss_conf.item())

        # TODO : the best lambda
        loss_xy *= 2.0
        # loss_wh *= 0.05
        # loss_cls *= 0.5
        # loss_conf *= 1.0

        # TODO : MEAN OR SUM ? For example : loss = loss * bs
        loss = loss_xy + loss_wh + loss_cls + loss_conf

        return loss