# -*- coding: utf-8 -*-
import torch
from utils import timer

from data import cfg

#@torch.jit.script
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    return torch.clamp(max_xy - min_xy, min=0).prod(3)  # inter


def jaccard(box_a, box_b, iscrowd:bool=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)

def mask_iou(masks_a, masks_b, iscrowd=False):
    """
    Computes the pariwise mask IoU between two sets of masks of size [a, h, w] and [b, h, w].
    The output is of size [a, b].

    Wait I thought this was "box_utils", why am I putting this in here?
    """

    masks_a = masks_a.view(masks_a.size(0), -1)
    masks_b = masks_b.view(masks_b.size(0), -1)

    intersection = masks_a @ masks_b.t()
    area_a = masks_a.sum(dim=1).unsqueeze(1)
    area_b = masks_b.sum(dim=1).unsqueeze(0)

    return intersection / (area_a + area_b - intersection) if not iscrowd else intersection / area_a


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1)) + x_max


@torch.jit.script
def sanitize_coordinates(_x1, _x2, img_size:int, padding:int=0, cast:bool=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1-padding, min=0)
    x2 = torch.clamp(x2+padding, max=img_size)

    return x1, x2


@torch.jit.script
def crop(masks, boxes, padding:int=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)
    
    masks_left  = rows >= x1.view(1, 1, -1)
    masks_right = rows <  x2.view(1, 1, -1)
    masks_up    = cols >= y1.view(1, 1, -1)
    masks_down  = cols <  y2.view(1, 1, -1)
    
    crop_mask = masks_left * masks_right * masks_up * masks_down
    
    return masks * crop_mask.float()
