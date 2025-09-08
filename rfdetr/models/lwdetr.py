# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
lwdetr.py
This module implements the LW-DETR model, which is a variant of DETR designed for object detection.
It includes the model architecture, loss functions, and post-processing steps.
"""
import copy
import math
from typing import Callable
import torch
import torch.nn.functional as F
from torch import nn

from rfdetr.util import box_ops
from rfdetr.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)

from rfdetr.models.backbone import build_backbone
from rfdetr.models.matcher import build_matcher
from rfdetr.models.transformer import build_transformer
from rfdetr.models.segmentation import DETRsegm, PostProcessSegm, PostProcessPanoptic, dice_loss, sigmoid_focal_loss

class LWDETR(nn.Module):
    """ This is the Group DETR v3 module that performs object detection """
    def __init__(self,
                 backbone,
                 transformer,
                 num_classes,
                 num_queries,
                 aux_loss=False,
                 group_detr=1,
                 two_stage=False,
                 lite_refpoint_refine=False,
                 bbox_reparam=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            group_detr: Number of groups to speed detr training. Default is 1.
            lite_refpoint_refine: TODO
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        query_dim=4
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, query_dim)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.group_detr = group_detr

        # iter update
        self.lite_refpoint_refine = lite_refpoint_refine
        if not self.lite_refpoint_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.transformer.decoder.bbox_embed = None

        self.bbox_reparam = bbox_reparam

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # two_stage
        self.two_stage = two_stage
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(group_detr)])
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(group_detr)])

        self._export = False

    def reinitialize_detection_head(self, out_dim: int, *, include_background: bool = True, map_old_to_new: dict | None = None, prior_prob: float = 0.01):
        """
        Reinitialize (or resize) the detection classification head.

        Args:
            out_dim: Desired output dimension for class logits.
                    If include_background=True, pass NUM_CLASSES (excl. bg) and we add +1 internally.
                    If include_background=False, pass the exact final dimension.
            include_background: Whether out_dim excludes the background class (we'll add +1).
            map_old_to_new: Optional mapping {old_class_idx -> new_class_idx} (excluding background).
                            When provided and dims allow, copies corresponding weights/biases.
            prior_prob: Focal-style bias init.

        Background handling:
            - We assume the last logit index is background.
        """
        import math, copy
        from torch import nn

        old_head: nn.Linear = self.class_embed
        old_out = old_head.out_features
        d_model = self.transformer.d_model

        final_out = (out_dim + 1) if include_background else out_dim
        assert final_out >= 2, "final_out must be >= 2 (at least 1 class + background)"

        # Create new head
        new_head = nn.Linear(d_model, final_out)

        # Init weights
        nn.init.xavier_uniform_(new_head.weight)
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        with torch.no_grad():
            new_head.bias.fill_(bias_value)

        # Try to transfer overlapping class weights (excluding background) if mapping provided
        if map_old_to_new is not None and old_out >= 2 and final_out >= 2:
            old_bg = old_out - 1
            new_bg = final_out - 1
            with torch.no_grad():
                # Always re-init bg bias; copy only foreground classes
                for o, n in map_old_to_new.items():
                    if 0 <= o < old_bg and 0 <= n < new_bg:
                        new_head.weight[n].copy_(old_head.weight[o])
                        new_head.bias[n].copy_(old_head.bias[o])
                # Leave background (new_bg) at the focal prior bias

        # Swap in
        del self.class_embed
        self.add_module("class_embed", new_head)

        # Two-stage encoder classification heads (if present)
        if getattr(self, "two_stage", False):
            if hasattr(self.transformer, "enc_out_class_embed"):
                del self.transformer.enc_out_class_embed
            self.transformer.add_module(
                "enc_out_class_embed",
                nn.ModuleList([copy.deepcopy(self.class_embed) for _ in range(self.group_detr)])
            )

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            # only use one group in inference
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight)

        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat(
                    [outputs_coord_cxcy, outputs_coord_wh], dim=-1
                )
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            outputs_class = self.class_embed(hs)

            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                cls_enc.append(cls_enc_gidx)
            cls_enc = torch.cat(cls_enc, dim=1)
            if hs is not None:
                out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
            else:
                out = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
        
        return out

    def forward_export(self, tensors):
        srcs, _, poss = self.backbone(tensors)
        # only use one group in inference
        refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
        query_feat_weight = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, None, poss, refpoint_embed_weight, query_feat_weight)

        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat(
                    [outputs_coord_cxcy, outputs_coord_wh], dim=-1
                )
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
            outputs_class = self.class_embed(hs)
        else:
            assert self.two_stage, "if not using decoder, two_stage must be True"
            outputs_class = self.transformer.enc_out_class_embed[0](hs_enc)
            outputs_coord = ref_enc
            
        return outputs_coord, outputs_class

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        """ """
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]
        for i in range(vit_encoder_num_layers):
            if hasattr(self.backbone[0].encoder, 'blocks'): # Not aimv2
                if hasattr(self.backbone[0].encoder.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.blocks[i].drop_path.drop_prob = dp_rates[i]
            else: # aimv2
                if hasattr(self.backbone[0].encoder.trunk.blocks[i].drop_path, 'drop_prob'):
                    self.backbone[0].encoder.trunk.blocks[i].drop_path.drop_prob = dp_rates[i]

    def update_dropout(self, drop_rate):
        for module in self.transformer.modules():
            if isinstance(module, nn.Dropout):
                module.p = drop_rate


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 focal_alpha,
                 losses,
                 group_detr=1,
                 sum_group_losses=False,
                 use_varifocal_loss=False,
                 use_position_supervised_loss=False,
                 ia_bce_loss=False,):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            group_detr: Number of groups to speed detr training. Default is 1.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.group_detr = group_detr
        self.sum_group_losses = sum_group_losses
        self.use_varifocal_loss = use_varifocal_loss
        self.use_position_supervised_loss = use_position_supervised_loss
        self.ia_bce_loss = ia_bce_loss

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if self.ia_bce_loss:
            alpha = self.focal_alpha
            gamma = 2 
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            prob = src_logits.sigmoid()
            #init positive weights and negative weights
            pos_weights = torch.zeros_like(src_logits)
            neg_weights =  prob ** gamma

            pos_ind = [i for i in idx]
            pos_ind.append(target_classes_o)
            pos_idx = tuple(pos_ind)

            t = prob[pos_idx].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()

            pos_weights[pos_idx] = t.to(pos_weights.dtype)
            neg_weights[pos_idx] = 1 - t.to(neg_weights.dtype)
            # a reformulation of the standard loss_ce = - pos_weights * prob.log() - neg_weights * (1 - prob).log()
            # with a focus on statistical stability by using fused logsigmoid
            loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
            loss_ce = loss_ce.sum() / num_boxes

        elif self.use_position_supervised_loss:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()
            # pos_ious_func = pos_ious ** 2
            pos_ious_func = pos_ious

            cls_iou_func_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind = [i for i in idx]
            pos_ind.append(target_classes_o)
            cls_iou_func_targets[tuple(pos_ind)] = pos_ious_func

            norm_cls_iou_func_targets = cls_iou_func_targets \
                / (cls_iou_func_targets.view(cls_iou_func_targets.shape[0], -1, 1).amax(1, True) + 1e-8)
            loss_ce = position_supervised_loss(src_logits, norm_cls_iou_func_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        elif self.use_varifocal_loss:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            iou_targets=torch.diag(box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_ious = iou_targets.clone().detach()

            cls_iou_targets = torch.zeros((src_logits.shape[0], src_logits.shape[1],self.num_classes),
                                        dtype=src_logits.dtype, device=src_logits.device)

            pos_ind = [i for i in idx]
            pos_ind.append(target_classes_o)
            cls_iou_targets[tuple(pos_ind)] = pos_ious
            loss_ce = sigmoid_varifocal_loss(src_logits, cls_iou_targets, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        else:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:,:,:-1]
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Focal + Dice loss on matched instance masks."""
        assert "pred_masks" in outputs, "pred_masks missing from model outputs"
        pred_masks = outputs["pred_masks"]  # [B,Q,Hm,Wm] or [B,Q,1,Hm,Wm]
        if pred_masks.dim() == 5:
            pred_masks = pred_masks.squeeze(2)  # -> [B,Q,Hm,Wm]

        # pick only the matched predictions (sum(M_i) across batch)
        b_idx, q_idx = self._get_src_permutation_idx(indices)   # shapes [sumM], [sumM]
        src_masks = pred_masks[b_idx, q_idx]                     # [sumM,Hm,Wm]

        # build the matched target mask tensor, resized to pred mask size
        tgt_masks_resized = []
        Hm, Wm = src_masks.shape[-2:]
        for ( _, tgt_ids), t in zip(indices, targets):
            if len(tgt_ids) == 0:
                continue
            # t["masks"]: [Ni,H,W] (Tensor or Mask), select matched, then resize to (Hm,Wm)
            m = t["masks"][tgt_ids].float()                      # [Mi,H,W]
            m = F.interpolate(m[:, None], size=(Hm, Wm), mode="nearest").squeeze(1)  # [Mi,Hm,Wm]
            tgt_masks_resized.append(m)

        if len(tgt_masks_resized) == 0:
            zero = src_masks.sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}

        tgt_masks = torch.cat(tgt_masks_resized, dim=0).to(src_masks.dtype)  # [sumM,Hm,Wm]

        # focal (BCE-style) + dice on logits
        loss_mask = sigmoid_focal_loss(src_masks, tgt_masks, num_boxes, alpha=self.focal_alpha, gamma=2)
        loss_dice = dice_loss(src_masks, tgt_masks, num_boxes)
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    # Aux decoder outputs don't carry pred_masks; skip mask losses here.
                    if loss == 'masks' and 'pred_masks' not in aux_outputs:
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def sigmoid_varifocal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    focal_weight = targets * (targets > 0.0).float() + \
            (1 - alpha) * (prob - targets).abs().pow(gamma) * \
            (targets <= 0.0).float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * focal_weight

    return loss.mean(1).sum() / num_boxes


def position_supervised_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * (torch.abs(targets - prob) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcess(nn.Module):
    """RT-DETR style post-process with Top-K selection.
       If outputs contain 'pred_masks', also returns per-detection masks."""
    def __init__(self, num_select=300, mask_threshold=0.5) -> None:
        super().__init__()
        self.num_select = num_select
        self.mask_threshold = mask_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """
        outputs:
          - pred_logits: [B, Q, C] (sigmoid space, multi-label)
          - pred_boxes:  [B, Q, 4] (cx,cy,w,h in [0,1])
          - (optional) pred_masks: [B, Q, Hm, Wm] or [B, Q, 1, Hm, Wm] (logits)
        target_sizes: [B, 2] (h, w) per image
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # ----- Top-K selection over queries x classes -----
        B, Q, C = out_logits.shape
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(B, -1), self.num_select, dim=1)
        scores = topk_values                                     # [B, K]
        topk_boxes = topk_indexes // C                           # [B, K] -> query indices
        labels = topk_indexes % C                                # [B, K] -> class indices

        # ----- Boxes to absolute coords -----
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)             # [B, Q, 4]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))  # [B, K, 4]
        img_h, img_w = target_sizes.unbind(1)                    # [B], [B]
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)   # [B, 4]
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        # ----- Optional: attach instance masks for the selected detections -----
        if 'pred_masks' in outputs:
            masks = outputs['pred_masks']                        # [B, Q, Hm, Wm] or [B, Q, 1, Hm, Wm]
            if masks.dim() == 5:
                masks = masks.squeeze(2)                         # -> [B, Q, Hm, Wm]
            Bm, Qm, Hm, Wm = masks.shape
            assert Bm == B and Qm == Q, "pred_masks must align with logits/boxes"

            # Advanced indexing to pick masks of selected queries
            batch_idx = torch.arange(B, device=masks.device)[:, None]   # [B,1]
            sel_masks = masks[batch_idx, topk_boxes]                    # [B, K, Hm, Wm]

            # Resize to each image size and binarize
            for i in range(B):
                mh, mw = int(img_h[i].item()), int(img_w[i].item())
                if sel_masks.shape[-2:] != (mh, mw):
                    up = F.interpolate(sel_masks[i].unsqueeze(1).float(), size=(mh, mw),
                                       mode="bilinear", align_corners=False).squeeze(1)  # [K, H, W]
                else:
                    up = sel_masks[i].float()
                bin_mask = (up.sigmoid() > self.mask_threshold).to(torch.uint8)           # [K, H, W]
                results[i]['masks'] = bin_mask

        return results



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _resolve_target_shape(args):
    # Prefer args.shape; else square from args.resolution; else default 640
    if hasattr(args, 'shape'):
        return args.shape
    if hasattr(args, 'resolution'):
        return (args.resolution, args.resolution)
    return (640, 640)


def _resolve_device(args):
    # Favor explicitly provided device; else CUDA if available, else CPU
    if hasattr(args, 'device'):
        return torch.device(args.device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(args):
    """
    Returns:
        model, criterion, postprocessors
        (criterion/postprocessors left as None to keep the signature consistent;
         plug yours in if/when needed)
    """
    # ---- Class count convention (background INCLUDED) ----
    # LWDETR expects background included; keep +1 (max_obj_id + 1)
    num_classes = args.num_classes + 1

    # ---- Build backbone ----
    target_shape = _resolve_target_shape(args)
    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=args.vit_encoder_num_layers,
        pretrained_encoder=args.pretrained_encoder,
        window_block_indexes=args.window_block_indexes,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=args.use_cls_token,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=target_shape,
        rms_norm=args.rms_norm,
        backbone_lora=args.backbone_lora,
        force_no_pretrain=args.force_no_pretrain,
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov2_weights=args.pretrain_weights is None,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
    )

    # ---- Early exits (keep tuple shape) ----
    if getattr(args, 'encoder_only', False):
        # Expect backbone to be (backbone_module, ...) with .encoder inside 0th item
        assert hasattr(backbone[0], 'encoder'), "backbone[0] must expose `.encoder` when encoder_only=True"
        return backbone[0].encoder, None, None

    if getattr(args, 'backbone_only', False):
        return backbone, None, None

    # ---- Assertions / derived args ----
    assert hasattr(args, 'projector_scale') and hasattr(args, 'out_feature_indexes'), \
        "args must have projector_scale and out_feature_indexes"
    assert len(args.projector_scale) == len(args.out_feature_indexes), \
        "projector_scale and out_feature_indexes must have the same length"
    assert getattr(args, 'num_queries', 0) > 0, "num_queries must be > 0"
    args.num_feature_levels = len(args.projector_scale)

    # ---- Transformer ----
    transformer = build_transformer(args)

    # ---- Detector ----
    model = LWDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
    )

    # ---- Optional segmentation head ----
    if getattr(args, 'masks', False):
        assert DETRsegm is not None, "segmentation.DETRsegm not found; check your imports/paths"
        freeze_flag = getattr(args, 'frozen_weights', None) is not None
        model = DETRsegm(model, freeze_detr=freeze_flag)

    # ---- Device placement ----
    device = _resolve_device(args)
    model = model.to(device)

    # Keep signature consistent; wire your criterion/postprocessors if desired
    return model, None, None

def build_criterion_and_postprocessors(args):
    device = torch.device(args.device)

    matcher = build_matcher(args)

    # weights
    weight_dict = {
        "loss_ce":   args.cls_loss_coef,     # <- was 1.0
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
    }
    losses = ["labels", "boxes", "cardinality"]

    # seg losses
    if getattr(args, "masks", False):
        weight_dict["loss_mask"] = getattr(args, "mask_loss_coef", 1.0)
        weight_dict["loss_dice"] = getattr(args, "dice_loss_coef", 1.0)
        losses.append("masks")

    # aux expansion (decoder layers + optional encoder head)
    if getattr(args, "aux_loss", False):
        aux = {}
        for i in range(max(getattr(args, "dec_layers", 1) - 1, 0)):
            for k, v in weight_dict.items():
                aux[f"{k}_{i}"] = v
        if getattr(args, "two_stage", False):
            for k, v in weight_dict.items():
                aux[f"{k}_enc"] = v
        weight_dict.update(aux)

    criterion = SetCriterion(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=getattr(args, "focal_alpha", 0.25),
        losses=losses,
        group_detr=getattr(args, "group_detr", 1),
        sum_group_losses=getattr(args, "sum_group_losses", False),
        use_varifocal_loss=getattr(args, "use_varifocal_loss", False),
        use_position_supervised_loss=getattr(args, "use_position_supervised_loss", False),
        ia_bce_loss=getattr(args, "ia_bce_loss", True),
    ).to(device)

    # postprocessors
    postprocessors = {
        "bbox": PostProcess(
            num_select=getattr(args, "num_select", 300),
            mask_threshold=getattr(args, "mask_threshold", 0.5),
        )
    }
    if getattr(args, "masks", False):
        postprocessors["segm"] = PostProcessSegm(
            threshold=getattr(args, "mask_threshold", 0.5)
        )
        criterion.weight_dict.update({
            "loss_mask": args.mask_loss_coef,
            "loss_dice": args.dice_loss_coef,
        })
        # optional panoptic hook
        if getattr(args, "dataset_file", "") == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return criterion, postprocessors

