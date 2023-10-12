""" Contains functions used to sanitize and prepare the output of Yolact. """


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

from data import cfg, mask_type, MEANS, STD, activation_func
from utils.augmentations import Resize
from utils import timer
from .box_utils import sanitize_coordinates
from .polar_utils import polar2mask, polar2poly, polar2convex

def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                visualize_lincomb=False, score_threshold=0, output_mask_poly=False, convex_poly=False):
    """
    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The lost of dicts that Detect outputs.
        - w: The real width of the image.
        - h: The real height of the image.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
        - polygons
        - convex_polygons
    """
    
    dets = det_output[batch_idx]
    net = dets['net']
    dets = dets['detection']

    if dets is None:
        if output_mask_poly:
            return [torch.Tensor()] * 7 
        else:
            return [torch.Tensor()] * 6

    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]
        
        if dets['score'].size(0) == 0:
            if output_mask_poly:
                return [torch.Tensor()] * 7
            else:
                return [torch.Tensor()] * 6

    # Actually extract everything from dets now
    classes = dets['class']
    boxes   = dets['box']
    scores  = dets['score']
    masks   = dets['mask']
    polygons = dets['polygons']
    centers = dets['points']
    center_scores = dets['centerness']

    # Compute factor for removing padding
    if not cfg.fixed_size:
        pad_h,pad_w = Resize.calc_rescale((h,w), cfg.max_size)
        _h = pad_h/cfg.max_size[0]
        _w = pad_w/cfg.max_size[0]
    else:
        _h,_w = 1,1

    if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
        # At this points masks is only the coefficients
        proto_data = dets['proto']
        
        # Test flag, do not upvote
        if cfg.mask_proto_debug:
            np.save('scripts/proto.npy', proto_data.cpu().numpy())
        
        if visualize_lincomb:
            display_lincomb(proto_data, masks)

        masks = proto_data @ masks.t()
        masks = cfg.mask_proto_mask_activation(masks)

        mask_h,mask_w,_ = masks.shape

        crop_poly = polar2mask(centers, polygons * (1+cfg.extend_factor), (mask_w, mask_h))

        if convex_poly:
            polygons=polar2convex(polygons)

        if output_mask_poly:
            masks_poly = polar2mask(centers, polygons, (mask_w, mask_h))
        
        poly_coords = polar2poly(centers, polygons)
        poly_coords /= torch.Tensor([_w,_h]) # Remove padding
        poly_coords *= torch.Tensor([w,h])

        # Crop masks before upsampling because you know why
        masks *= crop_poly

        # Permute into the correct output shape [num_dets, proto_h, proto_w]
        masks = masks.permute(2, 0, 1).contiguous()
        if output_mask_poly:
            masks_poly = masks_poly.permute(2, 0, 1).contiguous()

        if cfg.use_maskiou:
            with timer.env('maskiou_net'):                
                with torch.no_grad():
                    maskiou_p = net.maskiou_net(masks.unsqueeze(1))
                    maskiou_p = torch.gather(maskiou_p, dim=1, index=classes.unsqueeze(1)).squeeze(1)
                    if cfg.rescore_mask:
                        if cfg.rescore_bbox:
                            scores = scores * maskiou_p
                        else:
                            scores = [scores, scores * maskiou_p]

        # Scale masks up to the full image
        masks = F.interpolate(masks.unsqueeze(0), (int(h/_h), int(w/_w)), mode=interpolation_mode, align_corners=False).squeeze(0)
        masks = masks[:,:h,:w]
        if output_mask_poly:
            masks_poly = F.interpolate(masks_poly.float().unsqueeze(0), (int(h/_h), int(w/_w)), mode=interpolation_mode, align_corners=False).squeeze(0)
            masks_poly = masks_poly[:,:h,:w]
        # Binarize the masks
        masks.gt_(0.5)
        if output_mask_poly:
            masks_poly.gt_(0.5)

    boxes /= torch.Tensor([_w,_h,_w,_h]) # Remove padding
    
    #if cfg.extend_factor==1:
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = boxes.long()
    if cfg.extend_factor!=0:
        # Tentative boxes_minus + rapide
        profile_x = masks.any(dim=1).int()
        profile_y = masks.any(dim=2).int()
        boxes_extend = torch.stack([profile_x.argmax(dim=1),profile_y.argmax(dim=1),w-profile_x.flip(1).argmax(dim=1),h-profile_y.flip(1).argmax(dim=1)],axis=-1)
        
        boxes[:,:2] = torch.minimum(boxes[:,:2],boxes_extend[:,:2])
        boxes[:,2:] = torch.maximum(boxes[:,2:],boxes_extend[:,2:])

    if cfg.mask_type == mask_type.direct and cfg.eval_mask_branch:
        # Upscale masks
        full_masks = torch.zeros(masks.size(0), h, w)

        for jdx in range(masks.size(0)):
            x1, y1, x2, y2 = boxes[jdx, :]

            mask_w = x2 - x1
            mask_h = y2 - y1

            # Just in case
            if mask_w * mask_h <= 0 or mask_w < 0:
                continue
            
            mask = masks[jdx, :].view(1, 1, cfg.mask_size, cfg.mask_size)
            mask = F.interpolate(mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
            mask = mask.gt(0.5).float()
            full_masks[jdx, y1:y2, x1:x2] = mask
        
        masks = full_masks

    if output_mask_poly:
        return classes, scores, center_scores, boxes, masks, poly_coords, masks_poly
    else:
        return classes, scores, center_scores, boxes, masks, poly_coords

    


def undo_image_transformation(img, w, h):
    """
    Takes a transformed image tensor and returns a numpy ndarray that is untransformed.
    Arguments w and h are the original height and width of the image.
    """
    img_numpy = img.permute(1, 2, 0).cpu().numpy()
    img_numpy = img_numpy[:, :, (2, 1, 0)] # To BRG

    if cfg.backbone.transform.normalize:
        img_numpy = (img_numpy * np.array(STD) + np.array(MEANS)) / 255.0
    elif cfg.backbone.transform.subtract_means:
        img_numpy = (img_numpy / 255.0 + np.array(MEANS) / 255.0).astype(np.float32)
        
    img_numpy = img_numpy[:, :, (2, 1, 0)] # To RGB
    img_numpy = np.clip(img_numpy, 0, 1)

    return cv2.resize(img_numpy, (w,h))


def display_lincomb(proto_data, masks):
    out_masks = torch.matmul(proto_data, masks.t())
    # out_masks = cfg.mask_proto_mask_activation(out_masks)

    for kdx in range(1):
        jdx = kdx + 0
        import matplotlib.pyplot as plt
        coeffs = masks[jdx, :].cpu().numpy()
        idx = np.argsort(-np.abs(coeffs))
        # plt.bar(list(range(idx.shape[0])), coeffs[idx])
        # plt.show()
        
        coeffs_sort = coeffs[idx]
        arr_h, arr_w = (4,8)
        proto_h, proto_w, _ = proto_data.size()
        arr_img = np.zeros([proto_h*arr_h, proto_w*arr_w])
        arr_run = np.zeros([proto_h*arr_h, proto_w*arr_w])
        test = torch.sum(proto_data, -1).cpu().numpy()

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x

                if i == 0:
                    running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
                else:
                    running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

                running_total_nonlin = running_total
                if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                    running_total_nonlin = (1/(1+np.exp(-running_total_nonlin)))

                arr_img[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = (proto_data[:, :, idx[i]] / torch.max(proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
                arr_run[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = (running_total_nonlin > 0.5).astype(np.float)
        plt.imshow(arr_img)
        plt.show()

        plt.imshow(out_masks[:, :, jdx].cpu().numpy())
        plt.show()
