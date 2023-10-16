# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import log_sum_exp
from ..polar_utils import polar_target, detect_convex_indices, polar2mask
from data import cfg
import math
import cv2
import numpy as np
import os
import time

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, num_rays, negpos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.num_rays = num_rays
        self.negpos_ratio = negpos_ratio


    def forward(self, net, predictions, targets, masks, polygons, num_crowds):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            mask preds, and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_points,num_classes)
                masks shape: torch.size(batch_size,num_points,mask_dim)
                center shape: torch.size(batch_size,num_points)
                poly shape: torch.size(batch_size, num_points,num_rays)
                points shape: torch.size(num_points,2)
                
                proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)

            targets (list<tensor>): Ground truth boxes and labels for a batch, ###### ??????????
                shape: [batch_size][num_objs,5] (last idx is the label).

            masks (list<tensor>): Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]

            polygons (list<tensor>): Ground truth polygons for each object in each image,
                shape: [batch_size][num_objs,num_rays]

            num_crowds (list<int>): Number of crowd annotations per batch. The crowd
                annotations should be the last num_crowds elements of targets and masks.
            

        """
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        center_data = predictions['center'][:,:,0]
        poly_data = predictions['polygon']
        points = predictions['points']
        regress_ranges = predictions['regress_ranges']
        img_size = predictions['img_size']
        proto_data = predictions['proto']
       
        labels = [None] * len(targets) # Used in sem segm loss

        batch_size = conf_data.size(0)
        num_rays = poly_data.size(2)
        
 
        # Match points (default boxes) and ground truth boxes
        # These tensors will be created with the same device as conf_data       
        truths = []
        for idx in range(batch_size):
            truths.append(targets[idx][:, :-1].data)
            labels[idx] = targets[idx][:, -1].data.long()          

            # Split the crowd annotations because they come bundled in
            cur_crowds = num_crowds[idx]
            if cur_crowds > 0:
                split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
                crowd_boxes, truths[idx] = split(truths[idx])

                # We don't use the crowd labels or masks
                _, labels[idx] = split(labels[idx])
                _, masks[idx]  = split(masks[idx])
                _, polygons[idx] = split(polygons[idx])
            else:
                crowd_boxes = None
        bs_targets = [polar_target(truths[idx], polygons[idx], labels[idx], points, regress_ranges, num_rays, img_size, crowd_boxes, cfg.radius, cfg.force_gt_attribute) for idx in range(batch_size)]

        conf_t = Variable(torch.stack([bs_t[0] for bs_t in bs_targets],dim=0),requires_grad=False)
        poly_t = Variable(torch.stack([bs_t[2] for bs_t in bs_targets],dim=0),requires_grad=False)
        idx_t = Variable(torch.stack([bs_t[3] for bs_t in bs_targets],dim=0),requires_grad=False)

        # Centerness. If classic (PolarMask), computed from poly_target. Otherwise (ConvexMask), computed with ratio (dist to center) / (mec radius) 
        if cfg.polar_centerness :
            center_t = self.circular_centerness_target(poly_t)
        elif cfg.circular_centerness:
            center_t = Variable(torch.stack([bs_t[4] for bs_t in bs_targets],dim=0),requires_grad=False)
        else:
            raise NotImplementedError

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        
        # Shape: [batch,num_points,num_rays]
        pos_idx = pos.clone() 
        losses = {}
        
        ###############################################################
        # Create centerness targets from polygons targets
        weight = center_t[pos_idx].clone()
        avg_factor = weight.sum()

        # Polar Centerness loss
        if cfg.train_centerness:
            pos_ct = center_t>0
            num_pos_ct = pos_ct.sum(dim=1, keepdim=True)
            cent_t = center_t[pos_ct].view(-1)
            cent_p = center_data[pos_ct].view(-1)
            
            losses['N'] = -(torch.log(1-torch.abs(cent_p.sigmoid()-cent_t))).sum() * cfg.center_alpha


        # Mask loss
        if cfg.train_masks:
            loss = self.lincomb_mask_loss(pos, idx_t, poly_data, mask_data, points, proto_data, 
                                            masks, labels, center_t.clone())
            losses.update(loss)

        # Polar IoU Loss (Polygons)
        if cfg.train_polygons:
            poly_t = poly_t[pos_idx].view(-1,num_rays)
            poly_p = poly_data[pos_idx].view(-1,num_rays)
            mean_radius = poly_p.mean()/poly_t.mean()

            losses['P'] = self.polar_iou_loss(poly_p, poly_t, weight.clone()) * cfg.poly_alpha 

        # Confidence loss
        if cfg.use_focal_loss:
            losses['C'] = self.focal_conf_loss(conf_data, conf_t)
        else:
            losses['C'] = self.ohem_conf_loss(conf_data, conf_t, pos, batch_size)

        # These losses also don't depend on anchors

        if cfg.use_semantic_segmentation_loss:
            losses['S'] = self.semantic_segmentation_loss(predictions['segm'], masks, labels)

        # Divide all losses by the number of positives.
        # Don't do it for loss[P] because that doesn't depend on the anchors.
        total_num_pos = num_pos.data.sum().float()
        total_num_pos_ct = num_pos_ct.data.sum().float()
        for k in losses:
            if k not in ('S'):
                if k in ['P','M']:
                  losses[k] /= avg_factor
                elif k=='N':
                  losses[k] /= total_num_pos_ct
                else:
                  losses[k] /= total_num_pos
            else:
                losses[k] /= batch_size
        
        # Loss Key:
        #  - C: Class Confidence Loss
        #  - M: Mask Loss
        #  - P: Polygon Loss
        #  - N: Centerness Loss
        #  - S: Semantic Segmentation Loss

        return losses


    def circular_centerness_target(self, pos_mask_targets):
        centerness_targets = (pos_mask_targets.min(dim=-1)[0] / torch.clamp(pos_mask_targets.max(dim=-1)[0],1e-4))
        return torch.sqrt(centerness_targets)

    def polar_iou_loss(self, pred, target, weight=None):
        '''
         :param pred:  shape (N,num_rays)
         :param target: shape (N,num_rays)
         :weight: shape (N)
         :avg_factor: shape(1)
         :return: loss
         '''
        #print(pred.unique())
        #print(target.unique())
        total = torch.stack([pred,target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0]

        loss = (l_max.sum(dim=1) / l_min.sum(dim=1).clamp(1e-3)).log()
        if weight is not None:
            loss = loss * weight
        return loss.sum()
    
    def semantic_segmentation_loss(self, segment_data, mask_t, class_t, interpolation_mode='bilinear'):
        # Note num_classes here is without the background class so cfg.num_classes-1
        batch_size, num_classes, mask_h, mask_w = segment_data.size()
        loss_s = 0
        
        for idx in range(batch_size):
            cur_segment = segment_data[idx]
            cur_class_t = class_t[idx]

            with torch.no_grad():
                downsampled_masks = F.interpolate(mask_t[idx].unsqueeze(0), (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()
                
                # Construct Semantic Segmentation
                segment_t = torch.zeros_like(cur_segment, requires_grad=False)
                for obj_idx in range(downsampled_masks.size(0)):
                    segment_t[cur_class_t[obj_idx]] = torch.max(segment_t[cur_class_t[obj_idx]], downsampled_masks[obj_idx])
            
            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_t, reduction='sum')
        
        return loss_s / mask_h / mask_w * cfg.semantic_segmentation_alpha


    def ohem_conf_loss(self, conf_data, conf_t, pos, num):
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)

        loss_c = log_sum_exp(batch_conf) - batch_conf[:, 0]

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos]        = 0 # filter out pos boxes
        loss_c[conf_t < 0] = 0 # filter out neutrals (conf_t = -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        
        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg[pos]        = 0
        neg[conf_t < 0] = 0 # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='none')

        loss_c = loss_c.sum()
        
        return cfg.conf_alpha * loss_c
    
    def focal_conf_loss(self, conf_data, conf_t):
        """
        Focal loss but using sigmoid like the original paper.
        Note: To make things mesh easier, the network still predicts 81 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        """
        num_classes = conf_data.size(-1)

        conf_t = conf_t.view(-1) # [batch_size]
        conf_data = conf_data.view(-1, num_classes) # [batch_size, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0 # can't mask with -1, so filter that out

        # Compute a one-hot embedding of conf_t
        # From https://github.com/kuangliu/pytorch-retinanet/blob/master/utils.py
        conf_one_t = torch.eye(num_classes, device=conf_t.get_device())[conf_t]
        conf_pm_t  = conf_one_t * 2 - 1 # -1 if background, +1 if forground for specific class

        logpt = F.logsigmoid(conf_data * conf_pm_t) # note: 1 - sigmoid(x) = sigmoid(-x)
        pt    = logpt.exp()

        at = cfg.focal_loss_alpha * conf_one_t + (1 - cfg.focal_loss_alpha) * (1 - conf_one_t)
        at[..., 0] = 0 # Set alpha for the background class to 0 because sigmoid focal loss doesn't use it

        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
        loss = keep * loss.sum(dim=-1)

        return cfg.conf_alpha * loss.sum()
    
    def focal_conf_objectness_loss(self, conf_data, conf_t):
        """
        Instead of using softmax, use class[0] to be the objectness score and do sigmoid focal loss on that.
        Then for the rest of the classes, softmax them and apply CE for only the positive examples.

        If class[0] = 1 implies forground and class[0] = 0 implies background then you achieve something
        similar during test-time to softmax by setting class[1:] = softmax(class[1:]) * class[0] and invert class[0].
        """

        conf_t = conf_t.view(-1) # [batch_size]
        conf_data = conf_data.view(-1, conf_data.size(-1)) # [batch_size, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0 # so that gather doesn't drum up a fuss

        background = (conf_t == 0).float()
        at = (1 - cfg.focal_loss_alpha) * background + cfg.focal_loss_alpha * (1 - background)

        logpt = F.logsigmoid(conf_data[:, 0]) * (1 - background) + F.logsigmoid(-conf_data[:, 0]) * background
        pt    = logpt.exp()

        obj_loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt

        # All that was the objectiveness loss--now time for the class confidence loss
        pos_mask = conf_t > 0
        conf_data_pos = (conf_data[:, 1:])[pos_mask] # Now this has just 80 classes
        conf_t_pos    = conf_t[pos_mask] - 1         # So subtract 1 here

        class_loss = F.cross_entropy(conf_data_pos, conf_t_pos, reduction='sum')

        return cfg.conf_alpha * (class_loss + (obj_loss * keep).sum())
    


    def lincomb_mask_loss(self, pos, idx_t, poly_data, mask_data, points, proto_data, masks,
                          labels, _weight=None, interpolation_mode='bilinear'):
        mask_h = proto_data.size(1)
        mask_w = proto_data.size(2)
        
        loss_m = 0
        for idx in range(mask_data.size(0)): # BATCH SIZE
            # Ca traite les masks GT (c'est un peu le bordel, mais bon...)
            with torch.no_grad():
                downsampled_masks = F.interpolate(masks[idx].unsqueeze(0), (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()

                downsampled_masks = downsampled_masks.gt(0.5).float()

            cur_pos = pos[idx] # Coord pts à analyser dans img idx du batch
            pos_idx_t = idx_t[idx, cur_pos] # Indices des best GTs par rapport au pts dans img idx du batch
            if _weight is not None:
              cur_weight = _weight[idx, cur_pos]
            else:
              cur_weight = None

            pos_gt_points = points[cur_pos,:]
            # Note: this is in point-form
            pos_gt_poly_t = poly_data[idx,cur_pos,:] # (num_pts, num_rays)


            if pos_idx_t.size(0) == 0:
                continue
            proto_masks = proto_data[idx]
            proto_coef  = mask_data[idx, cur_pos, :]
           
            # If we have over the allowed number of masks, select a random sample
            old_num_pos = proto_coef.size(0)
            if old_num_pos > cfg.masks_to_train:
                perm = torch.randperm(proto_coef.size(0))
                select = perm[:cfg.masks_to_train]

                proto_coef = proto_coef[select, :]
                pos_idx_t  = pos_idx_t[select]
                
                pos_gt_poly_t = pos_gt_poly_t[select, :]
                pos_gt_points = pos_gt_points[select, :]
                cur_weight = cur_weight[select]

            num_pos = proto_coef.size(0)
            mask_t = downsampled_masks[:, :, pos_idx_t]     
            label_t = labels[idx][pos_idx_t]     

            # Size: [mask_h, mask_w, num_pos]
            pred_masks = proto_masks @ proto_coef.t()

            if cfg.mask_crop_convex:   
                convex_indices = detect_convex_indices(pos_gt_poly_t.clone())
                pos_gt_poly_t = pos_gt_poly_t * convex_indices

            h,w,_ = pred_masks.shape
            ind_pred_masks = polar2mask(pos_gt_points.clone(), pos_gt_poly_t.clone()*(1+cfg.extend_factor), (w,h))

            pred_masks[torch.logical_not(ind_pred_masks)] = -10

            mask_t[torch.logical_not(ind_pred_masks)] = 0 

            pre_loss = F.binary_cross_entropy_with_logits(pred_masks.clone(), mask_t, reduction='none') ###           
            
            polygon_area = torch.clamp(torch.sum(ind_pred_masks, dim=(0,1)),1) / (mask_w * mask_h) 
            pre_loss = pre_loss.sum(dim=(0,1)) / polygon_area
            
                
            # If the number of masks were limited scale the loss accordingly
            if old_num_pos > num_pos:
                pre_loss *= old_num_pos / num_pos

            if _weight is not None:
                pre_loss = pre_loss * cur_weight
            loss_m += torch.sum(pre_loss)
            

        losses = {'M': loss_m * cfg.mask_alpha / mask_h / mask_w}
        

        return losses


