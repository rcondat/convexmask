# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp, decode, center_size, crop, elemwise_mask_iou, elemwise_box_iou, crop_amp
from ..polar_utils import polar_target, detect_convex_indices, detect_convex_points, get_convex_rays, crop_boxes, polar2mask
from data import cfg, mask_type, activation_func
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

    def __init__(self, num_classes, num_rays, pos_threshold, neg_threshold, negpos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.num_rays = num_rays
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio

        # If you output a proto mask with this area, your l1 loss will be l1_alpha
        # Note that the area is relative (so 1 would be the entire image)
        self.l1_expected_area = 20*20/70/70
        self.l1_alpha = 0.1

        if cfg.use_class_balanced_conf:
            self.class_instances = None
            self.total_instances = 0

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
            
            * Only if mask_type == lincomb
        """
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        center_data = predictions['center'][:,:,0]
        poly_data = predictions['polygon']
        points = predictions['points']
        regress_ranges = predictions['regress_ranges']
        num_points_per_level = predictions['num_points']
        img_size = predictions['img_size']
        if cfg.mask_type == mask_type.lincomb:
            proto_data = predictions['proto']

        score_data = predictions['score'] if cfg.use_mask_scoring   else None   
        inst_data  = predictions['inst']  if cfg.use_instance_coeff else None
        
        labels = [None] * len(targets) # Used in sem segm loss

        batch_size = conf_data.size(0)
        num_points = points.size(0)
        num_rays = poly_data.size(2)
        num_classes = self.num_classes
        
        masks_shape = [m.shape[0] for m in masks]
 
        # Match points (default boxes) and ground truth boxes
        # These tensors will be created with the same device as conf_data
        if cfg.use_class_existence_loss:
            class_existence_t = conf_data.new(batch_size, num_classes-1)         
        truths = []
        for idx in range(batch_size):
            truths.append(targets[idx][:, :-1].data)
            labels[idx] = targets[idx][:, -1].data.long()          
            if cfg.use_class_existence_loss:
                # Construct a one-hot vector for each object and collapse it into an existence vector with max
                # Also it's fine to include the crowd annotations here
                class_existence_t[idx, :] = torch.eye(num_classes-1, device=conf_t.get_device())[labels[idx]].max(dim=0)[0]


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
        bs_targets = [polar_target(truths[idx], polygons[idx], labels[idx], points, regress_ranges, num_rays, num_points_per_level, img_size, crowd_boxes, cfg.inside_polygon, cfg.radius, cfg.force_gt_attribute) for idx in range(batch_size)]

        conf_t = Variable(torch.stack([bs_t[0] for bs_t in bs_targets],dim=0),requires_grad=False)
        loc_t = Variable(torch.stack([bs_t[1] for bs_t in bs_targets],dim=0),requires_grad=False)
        poly_t = Variable(torch.stack([bs_t[2] for bs_t in bs_targets],dim=0),requires_grad=False)
        idx_t = Variable(torch.stack([bs_t[3] for bs_t in bs_targets],dim=0),requires_grad=False)

        # Centerness. If classic (PolarMask), computed from poly_target. Otherwise (ConvexMask), computed with ratio (dist to center) / (mec radius) 
        if cfg.polar_centerness :
            center_t = self.polar_centerness_target(poly_t)
        elif cfg.mec_centerness:
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
        """
        path = '/home/2017018/rconda01/visu/'
        os.makedirs(path,exist_ok=True)

        os.makedirs(path+'centerness/',exist_ok=True)
       
        for i in range(center_t.shape[1]):
            if conf_t[0,i]>0:
                np.savetxt(path+'centerness/{}_{}.txt'.format(idx_t[0,i],i),torch.concatenate([points[i,0].reshape(1),
                                                                                 points[i,1].reshape(1),
                                                                                 center_t[0,i].reshape(1),
                                                                                 regress_ranges[i,1].reshape(1),
                                                                                 conf_t[0,i].reshape(1)]).cpu().numpy())

        os.makedirs(path+'polygons/', exist_ok=True)
        for i in range(center_t.shape[1]):
            if conf_t[0,i]>0:
                np.savetxt(path+'polygons/{}_{}.txt'.format(idx_t[0,i],i),poly_t[0,i].cpu().numpy())
        exit()
        """
        """
        with open('/home/2017018/rconda01/centerness.txt','w') as txt:
            for i in range(center_t.shape[1]):
                txt.write('{} - {} - {} - {} - {}\n'.format(points[i,0],points[i,1],center_t[0,i],regress_ranges[i,1],conf_t[0,i]))
        center_p = center_data.sigmoid()
        with open('/home/2017018/rconda01/centerness_pred.txt','w') as txt:
            for i in range(center_data.shape[1]):
                txt.write('{} - {} - {} - {}\n'.format(points[i,0],points[i,1],center_p[0,i],regress_ranges[i,1]))
        
        exit()
        """
        weight = center_t[pos_idx].clone()
        avg_factor = weight.sum()

        # Polar Centerness loss
        if cfg.train_centerness:
            pos_ct = center_t>0
            num_pos_ct = pos_ct.sum(dim=1, keepdim=True)
            cent_t = center_t[pos_ct].view(-1)
            cent_p = center_data[pos_ct].view(-1)
            if cfg.bce_loss_centerness:
                losses['N'] = F.binary_cross_entropy_with_logits(cent_p, cent_t, reduction='sum') * cfg.center_alpha
            elif cfg.custom_loss_centerness:
                losses['N'] = -(torch.log(1-torch.abs(cent_p.sigmoid()-cent_t))).sum() * cfg.center_alpha
            else:
                raise NotImplementedError

        # Mask loss
        if cfg.train_masks:
            # C'est pas direct
            if cfg.mask_type == mask_type.direct:
                if cfg.use_gt_bboxes:
                    pos_masks = []
                    for idx in range(batch_size):
                        pos_masks.append(masks[idx][idx_t[idx, pos[idx]]])
                    masks_t = torch.cat(pos_masks, 0)
                    masks_p = mask_data[pos, :].view(-1, cfg.mask_dim)
                    losses['M'] = F.binary_cross_entropy(torch.clamp(masks_p, 0, 1), masks_t, reduction='sum') * cfg.mask_alpha
                else:
                    losses['M'] = self.direct_mask_loss(pos_idx, idx_t, loc_data, mask_data, points, masks)
            # C'est lincomb
            elif cfg.mask_type == mask_type.lincomb:
                ret = self.lincomb_mask_loss(pos, idx_t, poly_data, mask_data, points, proto_data, 
                                             masks, poly_t.clone(), score_data, inst_data, labels, center_t.clone())
                if cfg.use_maskiou:
                    loss, maskiou_targets = ret
                else:
                    loss = ret
                losses.update(loss)

                if cfg.mask_proto_loss is not None:
                    if cfg.mask_proto_loss == 'l1':
                        losses['K'] = torch.mean(torch.abs(proto_data)) / self.l1_expected_area * self.l1_alpha
                    elif cfg.mask_proto_loss == 'disj':
                        losses['K'] = -torch.mean(torch.max(F.log_softmax(proto_data, dim=-1), dim=-1)[0])

        # Polar IoU Loss (Polygons)
        if cfg.train_polygons:
            poly_t = poly_t[pos_idx].view(-1,num_rays)
            poly_p = poly_data[pos_idx].view(-1,num_rays)
            mean_radius = poly_p.mean()/poly_t.mean()

            losses['P'] = self.polar_iou_loss(poly_p, poly_t, weight.clone()) * cfg.poly_alpha 

        # Confidence loss
        if cfg.use_focal_loss:
            if cfg.use_sigmoid_focal_loss:
                losses['C'] = self.focal_conf_sigmoid_loss(conf_data, conf_t)
            elif cfg.use_objectness_score:
                losses['C'] = self.focal_conf_objectness_loss(conf_data, conf_t)
            else:
                losses['C'] = self.focal_conf_loss(conf_data, conf_t)
        else:
            if cfg.use_objectness_score:
                losses['C'] = self.conf_objectness_loss(conf_data, conf_t, batch_size, loc_p, loc_t, priors)
            else:
                losses['C'] = self.ohem_conf_loss(conf_data, conf_t, pos, batch_size)

        # Mask IoU Loss
        if cfg.use_maskiou and maskiou_targets is not None:
            losses['I'] = self.mask_iou_loss(net, maskiou_targets)

        # These losses also don't depend on anchors
        if cfg.use_class_existence_loss:
            losses['E'] = self.class_existence_loss(predictions['classes'], class_existence_t)
        if cfg.use_semantic_segmentation_loss:
            losses['S'] = self.semantic_segmentation_loss(predictions['segm'], masks, labels)

        # Divide all losses by the number of positives.
        # Don't do it for loss[P] because that doesn't depend on the anchors.
        total_num_pos = num_pos.data.sum().float()
        total_num_pos_ct = num_pos_ct.data.sum().float()
        for k in losses:
            if k not in ('K', 'E', 'S'):
                if k in ['P','M']:
                  losses[k] /= avg_factor
                elif k=='N':
                  losses[k] /= total_num_pos_ct
                else:
                  losses[k] /= total_num_pos
            else:
                losses[k] /= batch_size

        # Compute radius mean
        losses['R'] = mean_radius
        
        # Loss Key:
        #  - B: Box Localization Loss
        #  - C: Class Confidence Loss
        #  - M: Mask Loss
        #  - P: Polygon Loss
        #  - N: Polar Centerness Loss
        #  - K: Prototype Loss
        #  - D: Coefficient Diversity Loss
        #  - E: Class Existence Loss
        #  - S: Semantic Segmentation Loss
        return losses


    def polar_centerness_target(self, pos_mask_targets):
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
    
    def class_existence_loss(self, class_data, class_existence_t):
        return cfg.class_existence_alpha * F.binary_cross_entropy_with_logits(class_data, class_existence_t, reduction='sum')

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
        if cfg.ohem_use_most_confident:
            # i.e. max(softmax) along classes > 0 
            batch_conf = F.softmax(batch_conf, dim=1)
            loss_c, _ = batch_conf[:, 1:].max(dim=1)
        else:
            # i.e. -softmax(class 0 confidence)
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

        if cfg.use_class_balanced_conf:
            # Lazy initialization
            if self.class_instances is None:
                self.class_instances = torch.zeros(self.num_classes, device=targets_weighted.device)
            
            classes, counts = targets_weighted.unique(return_counts=True)
            
            for _cls, _cnt in zip(classes.cpu().numpy(), counts.cpu().numpy()):
                self.class_instances[_cls] += _cnt

            self.total_instances += targets_weighted.size(0)

            weighting = 1 - (self.class_instances[targets_weighted] / self.total_instances)
            weighting = torch.clamp(weighting, min=1/self.num_classes)

            # If you do the math, the average weight of self.class_instances is this
            avg_weight = (self.num_classes - 1) / self.num_classes

            loss_c = (loss_c * weighting).sum() / avg_weight
        else:
            loss_c = loss_c.sum()
        
        return cfg.conf_alpha * loss_c

    def focal_conf_loss(self, conf_data, conf_t):
        """
        Focal loss as described in https://arxiv.org/pdf/1708.02002.pdf
        Adapted from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
        Note that this uses softmax and not the original sigmoid from the paper.
        """
        conf_t = conf_t.view(-1) # [batch_size*num_priors]
        conf_data = conf_data.view(-1, conf_data.size(-1)) # [batch_size*num_priors, num_classes]

        # Ignore neutral samples (class < 0)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0 # so that gather doesn't drum up a fuss

        logpt = F.log_softmax(conf_data, dim=-1)
        logpt = logpt.gather(1, conf_t.unsqueeze(-1))
        logpt = logpt.view(-1)
        pt    = logpt.exp()

        # I adapted the alpha_t calculation here from
        # https://github.com/pytorch/pytorch/blob/master/modules/detectron/softmax_focal_loss_op.cu
        # You'd think you want all the alphas to sum to one, but in the original implementation they
        # just give background an alpha of 1-alpha and each forground an alpha of alpha.
        background = (conf_t == 0).float()
        at = (1 - cfg.focal_loss_alpha) * background + cfg.focal_loss_alpha * (1 - background)

        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt

        # See comment above for keep
        return cfg.conf_alpha * (loss * keep).sum()
    
    def focal_conf_sigmoid_loss(self, conf_data, conf_t):
        """
        Focal loss but using sigmoid like the original paper.
        Note: To make things mesh easier, the network still predicts 81 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        """
        num_classes = conf_data.size(-1)

        conf_t = conf_t.view(-1) # [batch_size*num_priors]
        conf_data = conf_data.view(-1, num_classes) # [batch_size*num_priors, num_classes]

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

        conf_t = conf_t.view(-1) # [batch_size*num_priors]
        conf_data = conf_data.view(-1, conf_data.size(-1)) # [batch_size*num_priors, num_classes]

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
    
    def conf_objectness_loss(self, conf_data, conf_t, batch_size, loc_p, loc_t, priors):
        """
        Instead of using softmax, use class[0] to be p(obj) * p(IoU) as in YOLO.
        Then for the rest of the classes, softmax them and apply CE for only the positive examples.
        """

        conf_t = conf_t.view(-1) # [batch_size*num_priors]
        conf_data = conf_data.view(-1, conf_data.size(-1)) # [batch_size*num_priors, num_classes]

        pos_mask = (conf_t > 0)
        neg_mask = (conf_t == 0)

        obj_data = conf_data[:, 0]
        obj_data_pos = obj_data[pos_mask]
        obj_data_neg = obj_data[neg_mask]

        # Don't be confused, this is just binary cross entropy similified
        obj_neg_loss = - F.logsigmoid(-obj_data_neg).sum()

        with torch.no_grad():
            pos_priors = priors.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 4)[pos_mask, :]

            boxes_pred = decode(loc_p, pos_priors, cfg.use_yolo_regressors)
            boxes_targ = decode(loc_t, pos_priors, cfg.use_yolo_regressors)

            iou_targets = elemwise_box_iou(boxes_pred, boxes_targ)

        obj_pos_loss = - iou_targets * F.logsigmoid(obj_data_pos) - (1 - iou_targets) * F.logsigmoid(-obj_data_pos)
        obj_pos_loss = obj_pos_loss.sum()

        # All that was the objectiveness loss--now time for the class confidence loss
        conf_data_pos = (conf_data[:, 1:])[pos_mask] # Now this has just 80 classes
        conf_t_pos    = conf_t[pos_mask] - 1         # So subtract 1 here

        class_loss = F.cross_entropy(conf_data_pos, conf_t_pos, reduction='sum')

        return cfg.conf_alpha * (class_loss + obj_pos_loss + obj_neg_loss)


    def direct_mask_loss(self, pos_idx, idx_t, loc_data, mask_data, priors, masks):
        """ Crops the gt masks using the predicted bboxes, scales them down, and outputs the BCE loss. """
        loss_m = 0
        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                cur_pos_idx = pos_idx[idx, :, :]
                cur_pos_idx_squeezed = cur_pos_idx[:, 1]

                # Shape: [num_priors, 4], decoded predicted bboxes
                pos_bboxes = decode(loc_data[idx, :, :], priors.data, cfg.use_yolo_regressors)
                pos_bboxes = pos_bboxes[cur_pos_idx].view(-1, 4).clamp(0, 1)
                pos_lookup = idx_t[idx, cur_pos_idx_squeezed]

                cur_masks = masks[idx]
                pos_masks = cur_masks[pos_lookup, :, :]
                
                # Convert bboxes to absolute coordinates
                num_pos, img_height, img_width = pos_masks.size()

                # Take care of all the bad behavior that can be caused by out of bounds coordinates
                x1, x2 = sanitize_coordinates(pos_bboxes[:, 0], pos_bboxes[:, 2], img_width)
                y1, y2 = sanitize_coordinates(pos_bboxes[:, 1], pos_bboxes[:, 3], img_height)

                # Crop each gt mask with the predicted bbox and rescale to the predicted mask size
                # Note that each bounding box crop is a different size so I don't think we can vectorize this
                scaled_masks = []
                for jdx in range(num_pos):
                    tmp_mask = pos_masks[jdx, y1[jdx]:y2[jdx], x1[jdx]:x2[jdx]]

                    # Restore any dimensions we've left out because our bbox was 1px wide
                    while tmp_mask.dim() < 2:
                        tmp_mask = tmp_mask.unsqueeze(0)

                    new_mask = F.adaptive_avg_pool2d(tmp_mask.unsqueeze(0), cfg.mask_size)
                    scaled_masks.append(new_mask.view(1, -1))

                mask_t = torch.cat(scaled_masks, 0).gt(0.5).float() # Threshold downsampled mask
            
            pos_mask_data = mask_data[idx, cur_pos_idx_squeezed, :]
            loss_m += F.binary_cross_entropy(torch.clamp(pos_mask_data, 0, 1), mask_t, reduction='sum') * cfg.mask_alpha

        return loss_m
    

    def coeff_diversity_loss(self, coeffs, instance_t):
        """
        coeffs     should be size [num_pos, num_coeffs]
        instance_t should be size [num_pos] and be values from 0 to num_instances-1
        """
        num_pos = coeffs.size(0)
        instance_t = instance_t.view(-1) # juuuust to make sure

        coeffs_norm = F.normalize(coeffs, dim=1)
        cos_sim = coeffs_norm @ coeffs_norm.t()

        inst_eq = (instance_t[:, None].expand_as(cos_sim) == instance_t[None, :].expand_as(cos_sim)).float()

        # Rescale to be between 0 and 1
        cos_sim = (cos_sim + 1) / 2

        # If they're the same instance, use cosine distance, else use cosine similarity
        loss = (1 - cos_sim) * inst_eq + cos_sim * (1 - inst_eq)

        # Only divide by num_pos once because we're summing over a num_pos x num_pos tensor
        # and all the losses will be divided by num_pos at the end, so just one extra time.
        return cfg.mask_proto_coeff_diversity_alpha * loss.sum() / num_pos


    def lincomb_mask_loss(self, pos, idx_t, poly_data, mask_data, points, proto_data, masks,
                          gt_poly_t, score_data, inst_data, labels, _weight=None, interpolation_mode='bilinear'):
        mask_h = proto_data.size(1)
        mask_w = proto_data.size(2)
        num_rays = poly_data.size(-1)
        process_gt_polygons = cfg.mask_proto_normalize_emulate_roi_pooling or cfg.mask_proto_crop

        if cfg.mask_proto_remove_empty_masks:
            # Make sure to store a copy of this because we edit it to get rid of all-zero masks
            pos = pos.clone()

        loss_m = 0
        loss_d = 0 # Coefficient diversity loss

        maskiou_t_list = []
        maskiou_net_input_list = []
        label_t_list = []
        for idx in range(mask_data.size(0)): # BATCH SIZE
            # Ca traite les masks GT (c'est un peu le bordel, mais bon...)
            with torch.no_grad():
                downsampled_masks = F.interpolate(masks[idx].unsqueeze(0), (mask_h, mask_w),
                                                  mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()

                if cfg.mask_proto_binarize_downsampled_gt:
                    downsampled_masks = downsampled_masks.gt(0.5).float()

                if cfg.mask_proto_remove_empty_masks:
                    # Get rid of gt masks that are so small they get downsampled away
                    very_small_masks = (downsampled_masks.sum(dim=(0,1)) <= 0.0001)
                    for i in range(very_small_masks.size(0)):
                        if very_small_masks[i]:
                            pos[idx, idx_t[idx] == i] = 0

                if cfg.mask_proto_reweight_mask_loss:
                    # Ensure that the gt is binary
                    if not cfg.mask_proto_binarize_downsampled_gt:
                        bin_gt = downsampled_masks.gt(0.5).float()
                    else:
                        bin_gt = downsampled_masks

                    gt_foreground_norm = bin_gt     / (torch.sum(bin_gt,   dim=(0,1), keepdim=True) + 0.0001)
                    gt_background_norm = (1-bin_gt) / (torch.sum(1-bin_gt, dim=(0,1), keepdim=True) + 0.0001)

                    mask_reweighting   = gt_foreground_norm * cfg.mask_proto_reweight_coeff + gt_background_norm
                    mask_reweighting  *= mask_h * mask_w

            cur_pos = pos[idx] # Coord pts à analyser dans img idx du batch
            pos_idx_t = idx_t[idx, cur_pos] # Indices des best GTs par rapport au pts dans img idx du batch
            if _weight is not None:
              cur_weight = _weight[idx, cur_pos]
            else:
              cur_weight = None

            if process_gt_polygons:
                pos_gt_points = points[cur_pos,:]
                # Note: this is in point-form
                if cfg.mask_proto_crop_with_pred_box:
                    # Get pred boxes locations
                    pos_gt_poly_t = poly_data[idx,cur_pos,:] # (num_pts, num_rays)

                else:
                    # Get GT boxes locations
                    pos_gt_poly_t = gt_poly_t[idx,cur_pos,:] # (num_pts, num_rays)

            if pos_idx_t.size(0) == 0:
                continue
            proto_masks = proto_data[idx]
            proto_coef  = mask_data[idx, cur_pos, :]
            if cfg.use_mask_scoring:
                mask_scores = score_data[idx, cur_pos, :]

            if cfg.mask_proto_coeff_diversity_loss:
                if inst_data is not None:
                    div_coeffs = inst_data[idx, cur_pos, :]
                else:
                    div_coeffs = proto_coef

                loss_d += self.coeff_diversity_loss(div_coeffs, pos_idx_t)
            
            # If we have over the allowed number of masks, select a random sample
            old_num_pos = proto_coef.size(0)
            if old_num_pos > cfg.masks_to_train:
                perm = torch.randperm(proto_coef.size(0))
                select = perm[:cfg.masks_to_train]

                proto_coef = proto_coef[select, :]
                pos_idx_t  = pos_idx_t[select]
                
                if process_gt_polygons:
                    pos_gt_poly_t = pos_gt_poly_t[select, :]
                    pos_gt_points = pos_gt_points[select, :]
                    cur_weight = cur_weight[select]

                if cfg.use_mask_scoring:
                    mask_scores = mask_scores[select, :]

            num_pos = proto_coef.size(0)
            mask_t = downsampled_masks[:, :, pos_idx_t]     
            label_t = labels[idx][pos_idx_t]     

            # Size: [mask_h, mask_w, num_pos]
            pred_masks = proto_masks @ proto_coef.t()
            #pred_masks = cfg.mask_proto_mask_activation(pred_masks) ###

            if cfg.mask_proto_double_loss:
                if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                    pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), mask_t, reduction='sum')
                else:
                    pre_loss = F.smooth_l1_loss(pred_masks, mask_t, reduction='sum')
                
                loss_m += cfg.mask_proto_double_loss_alpha * pre_loss

            if cfg.mask_proto_crop:
                """
                output = pred_masks[:,:,0].cpu().detach().numpy()
                omin = output.min()
                omax = output.max()
                output = ((output-omin)/(omax-omin))*255
                print(cv2.imwrite('/home/2017018/rconda01/{}_bef.png'.format(idx),output.astype(np.uint8)))
                """

                if cfg.mask_crop_convex:   
                    convex_indices = detect_convex_indices(pos_gt_poly_t.clone())
                    pos_gt_poly_t = pos_gt_poly_t * convex_indices

                h,w,_ = pred_masks.shape
                ind_pred_masks = polar2mask(pos_gt_points.clone(), pos_gt_poly_t.clone()*(1+cfg.extend_factor), (w,h))

                pred_masks[torch.logical_not(ind_pred_masks)] = -10

                mask_t[torch.logical_not(ind_pred_masks)] = 0 

                """
                if torch.any(torch.sum(ind_pred_masks,dim=(0,1))==0):
                  ind_sum = torch.sum(ind_pred_masks,dim=(0,1))
                  ind_0 = torch.argmin(ind_sum)
                  print(pos_gt_poly_t[ind_0,:,:][:,convex_ind_poly_t[ind_0]])
                  
                  output = pred_masks[:,:,ind_0].cpu().detach().numpy()
                  omin = output[output!=-100].min()
                  omax = output[output!=-100].max()                
                  output[output!=-100] = ((output[output!=-100]-omin)/(omax-omin))*255
                  output[output==-100] = 0
                  print(output.dtype)
                  print(cv2.imwrite('../{}_aft.png'.format(idx),output.astype(np.uint8)))
                  
                  output = ind_pred_masks[:,:,ind_0].cpu().detach.numpy()
                  print(cv2.imwrite('../{}_mask_p.png'.format(idx),(output*255).astype(np.uint8)))
                  
                  output = mask_t[:,:,ind_0].cpu().detach().numpy()
                  print(cv2.imwrite('../{}_mask_t.png'.format(idx),(output*255).astype(np.uint8)))

                  exit()
                 
                for i in range(pred_masks.shape[2]):
                    #output = pred_masks[:,:,i].sigmoid().cpu().detach().numpy()
                    #print(cv2.imwrite('/home/2017018/rconda01/{}_pred.png'.format(i),(output*255).astype(np.uint8)))
                
                    output = ind_pred_masks[:,:,i].cpu().detach().numpy()
                    print(cv2.imwrite('/home/2017018/rconda01/{}_crop.png'.format(i),(output*255).astype(np.uint8)))

                    output = mask_t[:,:,i].cpu().detach().numpy()
                    print(cv2.imwrite('/home/2017018/rconda01/{}_mask_t.png'.format(i),(output*255).astype(np.uint8)))

                exit()
                """ 
            
            else:
                ind_pred_masks = torch.ones_like(pred_masks).bool()

            if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                pre_loss = F.binary_cross_entropy_with_logits(pred_masks.clone(), mask_t, reduction='none') ###
                """
                os.makedirs('/home/2017018/rconda01/visu_loss/',exist_ok=True)
                for i in range(pre_loss.shape[2]):
                    #output = pre_loss[:,:,i].cpu().detach().numpy()
                    #output = ((output>99)*255).astype(np.uint8)
                    #print(cv2.imwrite('/home/2017018/rconda01/visu_loss/{}.png'.format(i),output))              
                    mask_100 = pred_masks[:,:,i] == -100
                    print("LOSS 100 : {} - LOSS OTHER : {} - LOSS : {}".format(pre_loss[:,:,i][mask_100].sum(),pre_loss[:,:,i][torch.logical_not(mask_100)].sum(), pre_loss[:,:,i].sum()))
                    print()
                """
            else:
                pre_loss = F.smooth_l1_loss(pred_masks, mask_t, reduction='none')
            
            if cfg.mask_proto_normalize_mask_loss_by_sqrt_area:
                gt_area  = torch.sum(mask_t, dim=(0, 1), keepdim=True)
                pre_loss = pre_loss / (torch.sqrt(gt_area) + 0.0001)
            
            if cfg.mask_proto_reweight_mask_loss:
                pre_loss = pre_loss * mask_reweighting[:, :, pos_idx_t]
            
            if cfg.mask_proto_normalize_emulate_roi_pooling:

                polygon_area = torch.clamp(torch.sum(ind_pred_masks, dim=(0,1)),1) / (mask_w * mask_h) 
                pre_loss = pre_loss.sum(dim=(0,1)) / polygon_area
                
                
            # If the number of masks were limited scale the loss accordingly
            if old_num_pos > num_pos:
                pre_loss *= old_num_pos / num_pos

            if _weight is not None:
                pre_loss = pre_loss * cur_weight
            loss_m += torch.sum(pre_loss)
            if cfg.use_maskiou:
                if cfg.discard_mask_area > 0:
                    gt_mask_area = torch.sum(mask_t, dim=(0, 1))
                    select = gt_mask_area > cfg.discard_mask_area

                    if torch.sum(select) < 1:
                        continue

                    pos_gt_poly_t = pos_gt_poly_t[select, :]
                    pred_masks = pred_masks[:, :, select]
                    mask_t = mask_t[:, :, select]
                    label_t = label_t[select]

                maskiou_net_input = pred_masks.permute(2, 0, 1).contiguous().unsqueeze(1)
                pred_masks = pred_masks.gt(0.5).float()                
                maskiou_t = self._mask_iou(pred_masks, mask_t)
                
                maskiou_net_input_list.append(maskiou_net_input)
                maskiou_t_list.append(maskiou_t)
                label_t_list.append(label_t)

        losses = {'M': loss_m * cfg.mask_alpha / mask_h / mask_w}
        
        if cfg.mask_proto_coeff_diversity_loss:
            losses['D'] = loss_d

        if cfg.use_maskiou:
            # discard_mask_area discarded every mask in the batch, so nothing to do here
            if len(maskiou_t_list) == 0:
                return losses, None

            maskiou_t = torch.cat(maskiou_t_list)
            label_t = torch.cat(label_t_list)
            maskiou_net_input = torch.cat(maskiou_net_input_list)

            num_samples = maskiou_t.size(0)
            if cfg.maskious_to_train > 0 and num_samples > cfg.maskious_to_train:
                perm = torch.randperm(num_samples)
                select = perm[:cfg.masks_to_train]
                maskiou_t = maskiou_t[select]
                label_t = label_t[select]
                maskiou_net_input = maskiou_net_input[select]

            return losses, [maskiou_net_input, maskiou_t, label_t]

        return losses

    def _mask_iou(self, mask1, mask2):
        intersection = torch.sum(mask1*mask2, dim=(0, 1))
        area1 = torch.sum(mask1, dim=(0, 1))
        area2 = torch.sum(mask2, dim=(0, 1))
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret

    def mask_iou_loss(self, net, maskiou_targets):
        maskiou_net_input, maskiou_t, label_t = maskiou_targets

        maskiou_p = net.maskiou_net(maskiou_net_input)

        label_t = label_t[:, None]
        maskiou_p = torch.gather(maskiou_p, dim=1, index=label_t).view(-1)

        loss_i = F.smooth_l1_loss(maskiou_p, maskiou_t, reduction='sum')
        
        return loss_i * cfg.maskiou_alpha
