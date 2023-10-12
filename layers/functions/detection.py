import torch
import torch.nn.functional as F
from ..box_utils import decode, jaccard, index2d
from ..polar_utils import poly2bbox, polar2poly, get_convex_rays, detect_convex_indices, poly2polar, polar2mask, poly_iou, box_poly_iou
from utils import timer
import time
from data import cfg, mask_type

import numpy as np
import math

class Detect(object):
    """At test time, Detect is the final layer of Polar Yolact.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        
        self.use_cross_class_nms = False
        self.use_fast_nms = True


    def __call__(self, predictions, net):
        """
        Args:
            conf_data: (tensor) Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            center_data : (tensor) Centerness preds from centerness layers
                Shape: [batch, num_priors, 1]
            polygons_shape: (tensor) Rays preds from polygon layers
                Shape: [batch, num_priors, num_rays]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        conf_data  = predictions['conf']
        center_data = predictions['center']
        poly_data  = predictions['polygon']
        mask_data  = predictions['mask']
        points_data = predictions['points']

        proto_data = predictions['proto'] if 'proto' in predictions else None
        inst_data  = predictions['inst']  if 'inst'  in predictions else None

        out = []

        with timer.env('Detect'):
            batch_size = conf_data.size(0)
            num_points = points_data.size(0)

            conf_preds = conf_data.view(batch_size, num_points, self.num_classes).transpose(2, 1).contiguous() # (batch_size, num_classes, num_points)
            center_preds = center_data.view(batch_size, num_points, 1).transpose(2, 1).contiguous() # (batch_size, 1, num_points)

            for batch_idx in range(batch_size):
                # Replace decoded boxes by polygon reconstruction

                # For future NMS with polygons
                """
                decoded_convex_indices = detect_convex_points(poly_data[batch_idx])
                decoded_convex_polygons = get_convex_rays(poly_data[batch_idx],decoded_convex_indices)
                """

                decoded_polygons = polar2poly(points_data, poly_data[batch_idx].clone()) #, self.angles)
                decoded_boxes = poly2bbox(decoded_polygons)
                result = self.detect(batch_idx, conf_preds, decoded_boxes, mask_data, inst_data, center_preds, poly_data[batch_idx], points_data)
                # Postprocess polygons to get convex polygons (new key)
                #if result is not None:
                #  polar_polygons = result['polygons'] # polar format
                #  convex_indices = detect_convex_indices(polar_polygons.clone())
                #  convex_polygons = get_convex_rays(polar_polygons.clone(), convex_indices)

                #  result['convex_polygons'] = convex_polygons


                if result is not None and proto_data is not None:
                    result['proto'] = proto_data[batch_idx]

                out.append({'detection': result, 'net': net})

        return out


    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data, center_preds=None, decoded_polygons=None, points_data=None):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores, _ = torch.max(cur_scores, dim=0) # (1,num_points)
        keep = (conf_scores > self.conf_thresh)
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]

        if center_preds is not None:
            centerness = center_preds[batch_idx,:,keep]
        else:
            centerness = None
        if decoded_polygons is not None:
            polygons = decoded_polygons[keep, :] #(200,num_points)
        else:
            polygons = None

        if points_data is not None:
          points_data = points_data[keep, :]
        
        # Multiply scores by centerness
        if centerness is not None:
            scores *= (centerness+cfg.centerness_factor).clip(0,1)
        if inst_data is not None:
            inst = inst_data[batch_idx, keep, :]
        
        if scores.size(1) == 0:
            return None
        if self.use_fast_nms:
            if self.use_cross_class_nms:
                boxes, masks, classes, scores, centerness, polygons, points = self.cc_fast_nms(boxes, masks, scores, centerness, polygons, points_data, self.nms_thresh, self.top_k)
            else:
                boxes, masks, classes, scores, centerness, polygons, points = self.fast_nms(boxes, masks, scores, centerness, polygons, points_data, self.nms_thresh, self.top_k)
        else:
            boxes, masks, classes, scores, centerness, polygons = self.traditional_nms(boxes, masks, scores, self.nms_thresh, self.conf_thresh, centerness, polygons)

            if self.use_cross_class_nms:
                print('Warning: Cross Class Traditional NMS is not implemented.')

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores, 'centerness': centerness, 'polygons': polygons, 'points': points}


    def cc_fast_nms(self, boxes, masks, scores, centerness=None, polygons=None, points=None, iou_threshold:float=0.5, top_k:int=200):
        # Collapse all the classes into 1 
        scores, classes = scores.max(dim=0)

        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]

        boxes_idx = boxes[idx]

        if cfg.nms_poly:
            poly_masks = polar2mask(points[idx][None,...].clone(),polygons[idx][None,...].clone(),(cfg.max_size[0]//5,cfg.max_size[1]//5)).permute((2,3,0,1))
            iou = mask_iou(poly_masks,poly_masks)[0]
        else:
            # Compute the pairwise IoU between the boxes
            iou = jaccard(boxes_idx, boxes_idx)
        
        # Zero out the lower triangle of the cosine similarity matrix and diagonal
        iou.triu_(diagonal=1)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the IoU matrix along the columns, each column will represent the maximum IoU
        # between this element and every element with a higher score than this element.
        iou_max, _ = torch.max(iou, dim=0)

        # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
        # don't have a higher scoring box that would supress it in normal NMS.
        idx_out = idx[iou_max <= iou_threshold]
        if centerness is not None:
            centerness = centerness[:,idx_out]
        if polygons is not None:
            polygons = polygons[idx_out]
        if points is not None:
            points = points[idx_out]
        return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out], centerness, polygons, points

    def fast_nms(self, boxes, masks, scores, centerness=None, polygons=None, points=None, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        scores, idx = scores.sort(1, descending=True)
        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]
        num_classes, num_dets = idx.size()
        
        # Polygon NMS
        # Create poly_masks
        if cfg.nms_poly:
            iou = box_poly_iou(boxes,polygons,points,R=72,idx=idx)
            iou = iou + iou.transpose(0,1)
        else:
            iou = jaccard(boxes[None,...], boxes[None,...])[0]

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)
        if centerness is not None:
            centerness = centerness[:,idx.view(-1)].view(num_classes, num_dets)
        if polygons is not None:
            polygons = polygons[idx.view(-1), :].view(num_classes,num_dets,-1)
        if points is not None:
            points = points[None, idx.view(-1), :].reshape(num_classes, num_dets, 2)

        iou = iou[idx[:,:,None].expand(num_classes,num_dets,num_dets).reshape(-1),idx[:,None,:].expand(num_classes,num_dets,num_dets).reshape(-1)].reshape(num_classes, num_dets, num_dets)
        iou.triu_(diagonal=1)

        iou_max, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= iou_threshold)

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        if second_threshold:
            keep *= (scores > self.conf_thresh)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]
        if centerness is not None:
            centerness = centerness[keep]
        if polygons is not None:
            polygons = polygons[keep]
        if points is not None:
            points = points[keep]      
        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]
        if centerness is not None:
            centerness = centerness[idx]
        if polygons is not None:
            polygons = polygons[idx]
        if points is not None:
            points = points[idx]
        return boxes, masks, classes, scores, centerness, polygons, points

    def traditional_nms(self, boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05, centerness=None, polygons=None):
        import pyximport
        pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

        from utils.cython_nms import nms as cnms

        num_classes = scores.size(0)

        idx_lst = []
        cls_lst = []
        scr_lst = []
        
        max_size = torch.Tensor([cfg.max_size[0],cfg.max_size[1],cfg.max_size[0],cfg.max_size[1]])
        
        # Multiplying by max_size is necessary because of how cnms computes its area and intersections
        boxes = boxes * max_size

        for _cls in range(num_classes):
            cls_scores = scores[_cls, :]
            idx = torch.arange(cls_scores.size(0), device=boxes.device)

            if cls_scores.size(0) == 0:
                continue
            
            preds = torch.cat([boxes, cls_scores[:, None]], dim=1).cpu().numpy()
            keep = cnms(preds, iou_threshold)
            keep = torch.Tensor(keep, device=boxes.device).long()

            idx_lst.append(idx[keep])
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores[keep])
        
        idx     = torch.cat(idx_lst, dim=0)
        classes = torch.cat(cls_lst, dim=0)
        scores  = torch.cat(scr_lst, dim=0)

        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        idx = idx[idx2]
        classes = classes[idx2]
        if centerness is not None:
            centerness = centerness[idx]
        if polygons is not None:
            polygons = polygons[idx]
        # Undo the multiplication above
        return boxes[idx] / max_size, masks[idx], classes, scores, centerness, polygons
