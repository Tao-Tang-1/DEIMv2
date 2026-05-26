"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Optimized Version for Wheat Detection & Academic Publication
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision
import copy

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register


@register()
class DEIMCriterion(nn.Module):
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self,
                 matcher,
                 weight_dict,
                 losses,
                 alpha=0.2,
                 gamma=2.0,
                 num_classes=80,
                 reg_max=32,
                 boxes_weight_format=None,
                 share_matched_indices=False,
                 mal_alpha=None,
                 use_uni_set=True):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.reg_max = reg_max
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set

        # 内部缓存
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    # --------------------------------------------------------------------------
    # 核心损失函数修改
    # --------------------------------------------------------------------------

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        """ Varifocal Loss 增强版：通过高质量正样本强化分类分支 """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        # 提点：使用 ious^2 作为 target，增加对高 IoU 样本的判别度
        target_score_o[idx] = ious.pow(2.0).to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        return {'loss_vfl': loss.mean(1).sum() * src_logits.shape[1] / num_boxes}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        """ 创新点 1: QAM (Quality-Aligned Matching) Loss
            利用预测分类置信度与定位 IoU 的对齐程度作为辅助监督信号
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        pred_scores_all = F.sigmoid(src_logits)

        # 计算对齐分数：sqrt(IoU * cls_score)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        matched_pred_scores = pred_scores_all[idx].detach()
        matched_cls_scores = torch.gather(matched_pred_scores, 1, target_classes_o.unsqueeze(-1)).squeeze(-1)
        quality_alignment = torch.sqrt(ious * matched_cls_scores + 1e-8)

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = quality_alignment.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        # 动态背景抑制
        bg_weight_factor = 1.5 if self.mal_alpha is None else self.mal_alpha
        weight = bg_weight_factor * pred_scores_all.detach().pow(self.gamma) * (1 - target) + target

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        return {'loss_mal': loss.mean(1).sum() * src_logits.shape[1] / num_boxes}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """ DIoU Loss：增加中心点约束，提升密集小麦场景下的定位稳定性 """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        # L1 Regression
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # DIoU 实现
        b1 = box_cxcywh_to_xyxy(src_boxes)
        b2 = box_cxcywh_to_xyxy(target_boxes)
        ious = torch.diag(box_iou(b1, b2)[0])

        lt = torch.min(b1[:, :2], b2[:, :2])
        rb = torch.max(b1[:, 2:], b2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        c2 = (wh[:, 0] ** 2 + wh[:, 1] ** 2) + 1e-7 # 外接矩形对角线平方
        d2 = (src_boxes[:, 0] - target_boxes[:, 0]) ** 2 + (src_boxes[:, 1] - target_boxes[:, 1]) ** 2 # 中心距离平方

        loss_diou = 1 - ious + (d2 / c2)
        if boxes_weight is not None:
            loss_diou = loss_diou * boxes_weight
        losses['loss_giou'] = loss_diou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """ 创新点 2: EDR (Entropy-based Distribution Refinement)
            通过减小分布熵，强制模型学习更确定的物体边界。
        """
        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners_raw = outputs['pred_corners'][idx]
            pred_corners = pred_corners_raw.reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()

            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                    self.fgl_targets_dn = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                    self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                     self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            # 基础 FGL Loss (多模态分布优化)
            ious = torch.diag(box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            l_fgl = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            # EDR：分布熵惩罚项
            probs = F.softmax(pred_corners, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
            l_entropy = (entropy * weight_targets).sum() / num_boxes
            losses['loss_fgl'] = l_fgl + 0.1 * l_entropy # 熵权重定为 0.1

            # DDF 蒸馏逻辑（全量保留）
            if 'teacher_corners' in outputs:
                pred_corners_all = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners_all = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners_all, target_corners_all):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]
                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                        (F.log_softmax(pred_corners_all / T, dim=1), F.softmax(target_corners_all.detach() / T, dim=1))).sum(-1)

                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5

                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    # --------------------------------------------------------------------------
    # 工具函数与转发逻辑 (全量补充)
    # --------------------------------------------------------------------------

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """ 合并所有解码层的匹配结果，构建全局候选集合 """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        """ 基础 Focal Loss 逻辑 """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        return {'loss_focal': loss.mean(1).sum() * src_logits.shape[1] / num_boxes}

    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1
        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)
        if weight is not None:
            loss = loss * weight.float()
        return loss.sum() / avg_factor if avg_factor else loss.sum()

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}
        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        if self.boxes_weight_format == 'iou':
            iou = torch.diag(box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))[0])
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()
        return {'boxes_weight': iou} if loss == 'boxes' else {'values': iou} if loss in ('vfl', 'mal') else {}

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device).tile(dn_num_group)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), torch.zeros(0, dtype=torch.int64, device=device)))
        return dn_match_indices

    def forward(self, outputs, targets, epoch=0, **kwargs):
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        indices = self.matcher(outputs_without_aux, targets, epoch=epoch)['indices']
        self._clear_cache()

        # 辅助匹配逻辑 (Go-indices)
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs'] + ([outputs['pre_outputs']] if 'pre_outputs' in outputs else [])
            for aux_outputs in aux_outputs_list:
                idx_aux = self.matcher(aux_outputs, targets, epoch=epoch)['indices']
                cached_indices.append(idx_aux)
                indices_aux_list.append(idx_aux)
            for aux_outputs in outputs['enc_aux_outputs']:
                idx_enc = self.matcher(aux_outputs, targets, epoch=epoch)['indices']
                cached_indices_enc.append(idx_enc)
                indices_aux_list.append(idx_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            indices_go = indices
            num_boxes_go = 1.0 # 占位

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            use_uni = self.use_uni_set and (loss in ['boxes', 'local'])
            idx_in, n_in = (indices_go, num_boxes_go) if use_uni else (indices, num_boxes)
            meta = self.get_loss_meta_info(loss, outputs, targets, idx_in)
            l_dict = self.get_loss(loss, outputs, targets, idx_in, n_in, **meta)
            losses.update({k: v * self.weight_dict[k] for k, v in l_dict.items() if k in self.weight_dict})

        # Aux Layers
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    use_uni = self.use_uni_set and (loss in ['boxes', 'local'])
                    idx_in, n_in = (indices_go, num_boxes_go) if use_uni else (cached_indices[i], num_boxes)
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, idx_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, idx_in, n_in, **meta)
                    losses.update({k + f'_aux_{i}': v * self.weight_dict[k] for k, v in l_dict.items() if k in self.weight_dict})

        # Encoder Aux
        if 'enc_aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    use_uni = self.use_uni_set and (loss == 'boxes')
                    idx_in, n_in = (indices_go, num_boxes_go) if use_uni else (cached_indices_enc[i], num_boxes)
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, idx_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, idx_in, n_in, **meta)
                    losses.update({k + f'_enc_{i}': v * self.weight_dict[k] for k, v in l_dict.items() if k in self.weight_dict})

        # DN (Denoising)
        if 'dn_outputs' in outputs:
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_n_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:
                    aux_outputs['is_dn'], aux_outputs['up'], aux_outputs['reg_scale'] = True, outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_n_boxes, **meta)
                    losses.update({k + f'_dn_{i}': v * self.weight_dict[k] for k, v in l_dict.items() if k in self.weight_dict})

        return {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}