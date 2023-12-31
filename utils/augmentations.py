import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from math import sqrt
import time
from data import cfg, MEANS, STD
import os
import albumentations as A

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, masks=None, boxes=None, polygons=None, labels=None,times=None):
        for t in self.transforms:
            img, masks, boxes, polygons, labels, times = t(img, masks, boxes, polygons, labels, times)
        return img, masks, boxes, polygons, labels, times



class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, masks=None, boxes=None, polygons=None, labels=None):
        return self.lambd(img, masks, boxes, polygons, labels)


class ConvertFromInts(object):
    def __call__(self, image, masks=None, boxes=None, polygons=None, labels=None,times=None):
        return image.astype(np.float32), masks, boxes, polygons, labels, times



class ToAbsoluteCoords(object):
    def __call__(self, image, masks=None, boxes=None, polygons=None, labels=None,times=None):
        height, width, channels = image.shape
        if boxes is not None:
            boxes[:, 0] *= width
            boxes[:, 2] *= width
            boxes[:, 1] *= height
            boxes[:, 3] *= height

        if polygons is not None:
            not_points = polygons<0
            polygons[:, :, 0] *= width
            polygons[:, :, 1] *= height
            polygons[not_points] = -1
        return image, masks, boxes, polygons, labels, times


class ToPercentCoords(object):
    def __call__(self, image, masks=None, boxes=None, polygons=None, labels=None, times=None):
        height, width, channels = image.shape
        if boxes is not None:
            boxes[:, 0] /= width
            boxes[:, 2] /= width
            boxes[:, 1] /= height
            boxes[:, 3] /= height
        if polygons is not None:
            not_points = polygons<0
            polygons[:, :, 0] /= width
            polygons[:, :, 1] /= height
            polygons[not_points]=-1
        return image, masks, boxes, polygons, labels, times

class GeneratePolygons(object):
    def __init__(self,return_convex=True):
        self.return_convex=return_convex

    def __call__(self, image, masks=None, boxes=None, polygons=None, labels=None, times=None):
        _, height, width  = masks.shape
        num_annots = masks.shape[0]
        num_crowds = labels['num_crowds']
        crowd_mask = np.zeros(masks.shape[0], dtype=np.int32)

        if num_crowds > 0:
            crowd_mask[-num_crowds:] = 1

        valid_ids = np.argwhere(masks.sum(axis=(1,2))>0)[:,0].tolist()
        if self.return_convex:
            list_polygons = [cv2.convexHull(np.concatenate(list(cv2.findContours(masks[j,:,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]),axis=0))[:,0,:] for j in valid_ids]
        else:
            list_polygons = [np.concatenate(list(cv2.findContours(masks[j,:,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]),axis=0)[:,0,:] for j in valid_ids]
        num_pts = [poly.shape[0] for poly in list_polygons]
        if len(num_pts) == 0:
          max_pts = 0
        else:
          max_pts = max(num_pts)
        polygons = np.ones((len(valid_ids),max_pts,2))*-1
        for j in range(len(valid_ids)):
            polygons[j,:num_pts[j],:]=list_polygons[j]/np.array([width,height])
        masks = masks[valid_ids]
        boxes = boxes[valid_ids]
        labels['labels'] = labels['labels'][valid_ids]
        if num_crowds > 0:
          labels['num_crowds'] = np.sum(crowd_mask[valid_ids])
                
        return image, masks, boxes, polygons, labels, times


class AlbumentationsAugment(object):
    def __init__(self):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=[-0.1, 0.1], contrast_limit=[-0.1, 0.3], brightness_by_max=True),
        
            A.GaussNoise(p=0.2, var_limit=(10.0, 50.0), mean=0, per_channel=True),
            A.GlassBlur(p=0.1, sigma=0.6, max_delta=3, iterations=2, mode='fast'),
            A.ISONoise(p=0.2, color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
                        
            A.HueSaturationValue(p=0.3, sat_shift_limit=0.25, hue_shift_limit=0, val_shift_limit=0),
            A.MotionBlur(p=0.2, blur_limit=7),
            A.Perspective(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_ids'], min_visibility=0.1)) 

    def __call__(self, image, masks, boxes=None, polygons=None, labels=None, times=None):

        num_annots = len(labels['labels'])
        if len(masks)==0:
            _masks=None
        else:
            _masks=[masks[i] for i in range(num_annots)]
        boxes=[boxes[i] for i in range(num_annots)]
        """
        print("IMAGE")
        print(image)
        print("BOXES")
        print(boxes)
        print("MASKS")
        print(_masks.shape)
        print("LABELS")
        print(labels)
        """

        old_boxes = boxes.copy()
        old_masks = masks.copy()
        old_labels = labels.copy()

        for i in range(50):
            transformed = self.transform(image=image.astype(np.uint8),
                                         masks=_masks,
                                         bboxes=boxes,
                                         category_id=labels['labels'],
                                         bbox_ids=np.arange(len(boxes)))
            if len(transformed['bboxes'])==0:
                if i==49:
                    print("ERROR : 50 boucles passees et ça fonctionne pas !")
                    exit()
                continue
            break

        image = np.array(transformed['image'])
        boxes = np.stack([list(b) for b in transformed['bboxes']],axis=0)
        visible_ids = transformed['bbox_ids']
        if len(masks)==0:
            _masks = masks
        else:
            _masks = np.stack(transformed['masks'],axis=0)[visible_ids]
        labels['labels'] = transformed['category_id'][visible_ids]
        if labels['num_crowds']!=0:
            bbox_ids = transformed['bbox_ids']
            crowd_ids = np.arange(num_annots)[:-labels['num_crowds']]
            labels['num_crowd'] = sum([c in bbox_ids for c in crowd_ids])
        """
        print("IMAGE")
        print(old_image)
        print(image)
        print("BOXES")
        print(old_boxes)
        print(boxes)
        print("MASKS")
        print(old_masks)
        print(_masks)
        print("LABELS")
        print(old_labels)
        print(labels)
        exit()
        """

        return image.astype(np.float32), _masks, boxes, polygons, labels, times

class Pad(object):
    """
    Pads the image to the input width and height, filling the
    background with mean and putting the image in the top-left.

    Note: this expects im_w <= width and im_h <= height
    """
    def __init__(self, max_size, mean=MEANS, pad_gt=True):
        self.mean = mean
        self.width, self.height = max_size[0], max_size[0]
        self.pad_gt = pad_gt

    def __call__(self, image, masks, boxes=None, polygons=None, labels=None,times=None):
        im_h, im_w, depth = image.shape

        expand_image = np.zeros(
            (self.height, self.width, depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[:im_h, :im_w] = image

        if self.pad_gt:
            expand_masks = np.zeros(
                (masks.shape[0], self.height, self.width),
                dtype=masks.dtype)
            expand_masks[:,:im_h,:im_w] = masks
            masks = expand_masks
        return expand_image, masks, boxes, polygons, labels, times

class RandomChoiceResize(object):

    def __init__(self, max_shape, scales, resize_gt=True):
        self.max_shape = max_shape
        self.scales = scales
        self.resize = Resize(max_shape, resize_gt)

    def __call__(self, image, masks, boxes, polygons=None, labels=None, times=None):
        scale_idx = np.random.randint(len(self.scales))
        target_shape = self.scales[scale_idx]
        return self.resize(image, masks, boxes, polygons, labels, times, target_shape=target_shape)
        

class Resize(object):

    @staticmethod
    def calc_size_preserve_ar(img_w, img_h, max_size):
        """ I mathed this one out on the piece of paper. Resulting width*height = approx max_size^2 """
        ratio = sqrt(img_w / img_h)
        w = max_size * ratio
        h = max_size / ratio
        return int(w), int(h)
    
    @staticmethod
    def calc_rescale(img_shape, out_shape):
        out_long, out_short = out_shape
        img_long = max(img_shape)
        img_short = min(img_shape)
        ratio = min(out_short/img_short,out_long/img_long)
        return (round(img_shape[0]*ratio),round(img_shape[1]*ratio))

    def __init__(self, max_shape, resize_gt=True):
        self.resize_gt = resize_gt
        self.max_shape = max_shape

    def __call__(self, image, masks, boxes, polygons=None, labels=None, times=None, target_shape=None):

        img_h, img_w, _ = image.shape
        
        if target_shape is None:
            target_shape = self.max_shape
        """
        if self.keep_ratio:
            width, height = Resize.calc_rescale((img_w,img_h), target_shape)
        else:
            width, height = target_shape
        """
        width, height = Resize.calc_rescale((img_w,img_h), target_shape)
        if width != img_w or height != img_h:
            image = cv2.resize(image, (width, height))

            if self.resize_gt:
                # Act like each object is a color channel
                masks = masks.transpose((1, 2, 0))
                masks = cv2.resize(masks, (width, height))
            
                # OpenCV resizes a (w,h,1) array to (s,s), so fix that
                if len(masks.shape) == 2:
                    masks = np.expand_dims(masks, 0)
                else:
                    masks = masks.transpose((2, 0, 1))

                # Scale bounding boxes (which are currently absolute coordinates)
                boxes[:, [0, 2]] *= (width  / img_w)
                boxes[:, [1, 3]] *= (height / img_h)
                if polygons is not None:
                    not_points = polygons<0
                    polygons[:,:,0] *= (width / img_w)
                    polygons[:,:,1] *= (height / img_h)
                    polygons[not_points] = -1


        # Discard boxes that are smaller than we'd like
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        keep = (w > cfg.discard_box_width) * (h > cfg.discard_box_height)
        try:
            masks = masks[keep]
            boxes = boxes[keep]
        except:
            print(masks.shape)
            print(keep)
            print(labels)
            print(boxes.shape)
            exit()
        if polygons is not None:
            polygons = polygons[keep]
        labels['labels'] = labels['labels'][keep]
        labels['num_crowds'] = (labels['labels'] < 0).sum()

        return image, masks, boxes, polygons, labels, times


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, masks=None, boxes=None, polygons=None, labels=None, times=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, masks, boxes, polygons, labels, times


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, polygons=None, labels=None, times=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, masks, boxes, polygons, labels, times


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, masks=None, boxes=None, polygons=None, labels=None, times=None):
        # Don't shuffle the channels please, why would you do this

        # if random.randint(2):
        #     swap = self.perms[random.randint(len(self.perms))]
        #     shuffle = SwapChannels(swap)  # shuffle channels
        #     image = shuffle(image)
        return image, masks, boxes, polygons, labels, times


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, masks=None, boxes=None, polygons=None, labels=None, times=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, masks, boxes, polygons, labels, times


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, masks=None, boxes=None, polygons=None, labels=None, times=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, masks, boxes, polygons, labels, times


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, masks=None, boxes=None, polygons=None, labels=None,times=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, masks, boxes, polygons, labels, times


class ToCV2Image(object):
    def __call__(self, tensor, masks=None, boxes=None, polygons=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), masks, boxes, polygons, labels


class ToTensor(object):
    def __call__(self, cvimage, masks=None, boxes=None, polygons=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), masks, boxes, polygons, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, masks, boxes=None, polygons=None, labels=None,times=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, masks, boxes, polygons, labels, times

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # This piece of code is bugged and does nothing:
                # https://github.com/amdegroot/ssd.pytorch/issues/68
                #
                # However, when I fixed it with overlap.max() < min_iou,
                # it cut the mAP in half (after 8k iterations). So it stays.
                #
                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # [0 ... 0 for num_gt and then 1 ... 1 for num_crowds]
                num_crowds = labels['num_crowds']
                crowd_mask = np.zeros(mask.shape, dtype=np.int32)

                if num_crowds > 0:
                    crowd_mask[-num_crowds:] = 1

                # have any valid boxes? try again if not
                # Also make sure you have at least one regular gt
                if not mask.any() or np.sum(1-crowd_mask[mask]) == 0:
                    continue

                # take only the matching gt masks
                current_masks = masks[mask, :, :].copy()

                # crop the current masks to the same dimensions as the image
                current_masks = current_masks[:, rect[1]:rect[3], rect[0]:rect[2]]

                if current_masks.sum()==0:
                   continue


                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()
                
                # take only matching gt polygons
                if polygons is not None:
                    current_polygons = polygons[mask, :, :].copy()
                else:
                    current_polygons = None

                # take only matching gt labels
                labels['labels'] = labels['labels'][mask]
                current_labels = labels

                # We now might have fewer crowd annotations
                if num_crowds > 0:
                    labels['num_crowds'] = np.sum(crowd_mask[mask])

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                
                if current_polygons is not None:
                    not_points = current_polygons<0
                    current_polygons = np.minimum(np.maximum(current_polygons,rect[:2]),rect[2:])
                    current_polygons -= rect[:2]
                    current_polygons[not_points] = -1

                return current_image, current_masks, current_boxes, current_polygons, current_labels, times


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, masks, boxes, polygons, labels, times):
        if random.randint(2):
            return image, masks, boxes, polygons, labels, times

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        expand_masks = np.zeros(
            (masks.shape[0], int(height*ratio), int(width*ratio)),
            dtype=masks.dtype)
        expand_masks[:,int(top):int(top + height),
                       int(left):int(left + width)] = masks
        masks = expand_masks

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        if polygons is not None:
            not_points = polygons<0
            polygons += (int(left), int(top))
            polygons[not_points] = -1
        return image, masks, boxes, polygons, labels, times


class RandomMirror(object):
    def __call__(self, image, masks, boxes, polygons, labels, times):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            masks = masks[:, :, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
            if polygons is not None:
                not_points = polygons<0
                polygons[:,:,0] = width - polygons[:,:,0]
                polygons[not_points] = -1
        return image, masks, boxes, polygons, labels, times


class RandomFlip(object):
    def __call__(self, image, masks, boxes, polygons, labels, times):
        height , _ , _ = image.shape
        if random.randint(2):
            image = image[::-1, :]
            masks = masks[:, ::-1, :]
            boxes = boxes.copy()
            boxes[:, 1::2] = height - boxes[:, 3::-2]
            if polygons is not None:
                not_points = polygons<0
                polygons[:,:,1] = height - polygons[:,:,1]
                polygons[not_points] = -1
        return image, masks, boxes, polygons, labels, times


class RandomRot90(object):
    def __call__(self, image, masks, boxes, polygons=None, labels=None, times=None):
        old_height , old_width , _ = image.shape
        k = random.randint(4)
        image = np.rot90(image,k)
        masks = np.array([np.rot90(mask,k) for mask in masks])
        boxes = boxes.copy()
        for _ in range(k):
            boxes = np.array([[box[1], old_width - 1 - box[2], box[3], old_width - 1 - box[0]] for box in boxes])
            # Je ne sais pas si c'est correct pour les polygons, mais je n'utilise pas vraiment la fonction... donc bon...
            if polygons is not None:
                not_points = polygons<0
                polygons = np.stack([polygons[:,:,1],old_width-1-polygons[:,:,0]],axis=2)
                polygons[not_points] = -1
            old_width, old_height = old_height, old_width
        return image, masks, boxes, polygons, labels, times


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, masks, boxes, polygons, labels, times):
        im = image.copy()
        im, masks, boxes, polygons, labels, times = self.rand_brightness(im, masks, boxes, polygons, labels, times)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, masks, boxes, polygons, labels, times = distort(im, masks, boxes, polygons, labels, times)
        return self.rand_light_noise(im, masks, boxes, polygons, labels, times)


class BackboneTransform(object):
    """
    Transforms a BRG image made of floats in the range [0, 255] to whatever
    input the current backbone network needs.

    transform is a transform config object (see config.py).
    in_channel_order is probably 'BGR' but you do you, kid.
    """
    def __init__(self, transform, mean, std, in_channel_order, img_size):
        self.mean = mean
        self.std  = std
        self.transform = transform

        # Here I use "Algorithms and Coding" to convert string permutations to numbers
        self.channel_map = {c: idx for idx, c in enumerate(in_channel_order)}
        self.channel_permutation = [self.channel_map[c] for c in transform.channel_order]

    def __call__(self, img, masks=None, boxes=None, polygons=None, labels=None,times=None):
        mean = np.tile(self.mean, (img.shape[0], img.shape[1], 1)).astype(np.float32)
        std = np.tile(self.std, (img.shape[0], img.shape[1], 1)).astype(np.float32)
        if self.transform.normalize:
            img = (img.astype(np.float32) - mean) / std
        elif self.transform.subtract_means:
            img = (img.astype(np.float32) - mean)
        elif self.transform.to_float:
            img = img / 255.
        else:
            img = img.astype(np.float32)
        #if self.transform.channel_order!='RGB':
        img = img[:, :, self.channel_permutation]
        return img.astype(np.float32), masks, boxes, polygons, labels, times




class BaseTransform(object):
    """ Transorm to be used when evaluating. """

    def __init__(self, mean=MEANS, std=STD, resize_gt=False):
        self.augment = Compose([
            ConvertFromInts(),
            enable_if(resize_gt,ToAbsoluteCoords()),
            Resize(cfg.max_size, resize_gt=resize_gt),
            enable_if(not cfg.fixed_size, Pad(cfg.max_size, mean, pad_gt=resize_gt)),
            enable_if(resize_gt,ToPercentCoords()),
            GeneratePolygons(),
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR',cfg.max_size)
        ])

    def __call__(self, img, masks=None, boxes=None, polygons=None, labels=None, times=None):
        return self.augment(img, masks, boxes, polygons, labels, times)

import torch.nn.functional as F

class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't suppport a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self):
        super().__init__()

        self.mean = torch.Tensor(MEANS).float().cuda()[None, :, None, None]
        self.std  = torch.Tensor( STD ).float().cuda()[None, :, None, None]
        self.transform = cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std  = self.std.to(img.device)
         
        # img assumed to be a pytorch BGR image with channel order [n, h, w, c]
        _, h, w, _ = img.size()
        img_size = Resize.calc_rescale((w, h), cfg.max_size)
        img_size = (img_size[1], img_size[0]) # Pytorch needs h, w
        #else:
        #img_size = cfg.max_size
        
        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)

        if not cfg.fixed_size:
            img_pad = img.new_zeros(1,3,cfg.max_size[0],cfg.max_size[0])*self.mean
            img_pad[:,:,:img_size[0],:img_size[1]]=img
            img = img_pad

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255
        
        if self.transform.channel_order != 'RGB':
            raise NotImplementedError
        
        img = img[:, (2, 1, 0), :, :].contiguous()

        # Return value is in channel order [n, c, h, w] and RGB
        return img

def do_nothing(img=None, masks=None, boxes=None, polygons=None, labels=None, times=None):
    return img, masks, boxes, polygons, labels, times


def enable_if(condition, obj):
    return obj if condition else do_nothing

class SSDAugmentation(object):
    """ Transform to be used when training. """

    def __init__(self, mean=MEANS, std=STD):
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            enable_if(cfg.albu_augment, AlbumentationsAugment()),
            enable_if(cfg.augment_photometric_distort, PhotometricDistort()),
            enable_if(cfg.augment_expand, Expand(mean)),
            enable_if(cfg.augment_random_sample_crop, RandomSampleCrop()),
            enable_if(cfg.augment_random_mirror, RandomMirror()),
            enable_if(cfg.augment_random_flip, RandomFlip()),
            enable_if(cfg.augment_resize,RandomChoiceResize(cfg.max_size,cfg.resize_scales)),
            enable_if(not cfg.augment_resize,Resize(cfg.max_size)),
            enable_if(not cfg.fixed_size, Pad(cfg.max_size, mean)),
            ToPercentCoords(),
            GeneratePolygons(), 
            BackboneTransform(cfg.backbone.transform, mean, std, 'BGR',cfg.max_size)
        ])

    def __call__(self, img, masks, boxes, polygons, labels,times):
        return self.augment(img, masks, boxes, polygons, labels, times)
