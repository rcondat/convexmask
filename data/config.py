from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone
import math

# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)

# for making bounding boxes pretty
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))


COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


COCO_RESIZE_SCALES = {
    400: [(660, 400), (660, 384), (660, 368), (660, 352),
          (660, 336), (660, 320)],
    600: [(1000, 600), (1000, 576), (1000, 552), (1000, 528),
          (1000, 504), (1000, 480)],
    800: [(1330, 800), (1330, 768), (1330, 736), (1330, 704),
          (1330, 672), (1330, 640)],
}

SYNTH_CLASSES = ('tree')

SYNTH_LABEL_MAP = {1: 1}

SYNTH_RESIZE_SCALES = {
    720: [(1280, 720), (1216, 684), (1152, 648), (1088, 612),
          (1024, 576), (960, 540)],
}


# ----------------------- CONFIG CLASS ----------------------- #

class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))
        
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)



# ----------------------- DATASETS ----------------------- #

dataset_base = Config({
    'name': 'Base Dataset',

    # Training images and annotations
    'train_images': './data/coco/images/',
    'train_info':   'path_to_annotation_file',

    # Validation images and annotations.
    'valid_images': './data/coco/images/',
    'valid_info':   'path_to_annotation_file',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,

    # A list of names for each of you classes.
    'class_names': COCO_CLASSES,

    # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
    # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
    # If not specified, this just assumes category ids start at 1 and increase sequentially.
    'label_map': None
})

synthtree43k_dataset = dataset_base.copy({
    'name': 'SynthTree43K',

    'train_images': './data/synthtree43k/train_set/',
    'train_info': './data/synthtree43k/annotations/train_RGB_convex.json',

    'valid_images': './data/synthtree43k/val_set/',
    'valid_info': './data/synthtree43k/annotations/val_RGB_convex.json',

    'test_images': './data/synthtree43k/test_set/',
    'test_info': './data/synthtree43k/annotations/test_RGB_convex.json',

 'class_names':('tree'),

})


coco2017_dataset = dataset_base.copy({
    'name': 'COCO 2017',
    'train_images': '/save/2017018/rconda01/COCO/images/train2017/',
    'train_info': '/save/2017018/rconda01/COCO/annotations/instances_train2017.json',

    'valid_images': '/save/2017018/rconda01/COCO/images/val2017/',
    'valid_info': '/save/2017018/rconda01/COCO/annotations/instances_val2017.json',

    'test_images': '/save/2017018/rconda01/COCO/images/test2017/',
    'test_info': '/save/2017018/rconda01/COCO/annotations/image_info_test-dev2017.json',

    'label_map': COCO_LABEL_MAP
})

coco2017_testdev_dataset = dataset_base.copy({
    'name': 'COCO 2017 Test-Dev',

    'valid_info': './data/coco/annotations/image_info_test-dev2017.json',
    'has_gt': False,

    'label_map': COCO_LABEL_MAP
})


# ----------------------- TRANSFORMS ----------------------- #

resnet_transform = Config({
    'channel_order': 'RGB',
    'normalize': True,
    'subtract_means': False,
    'to_float': False,
})

vgg_transform = Config({
    # Note that though vgg is traditionally BGR,
    # the channel order of vgg_reducedfc.pth is RGB.
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': True,
    'to_float': False,
})

darknet_transform = Config({
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': False,
    'to_float': True,
})





# ----------------------- BACKBONES ----------------------- #

backbone_base = Config({
    'name': 'Base Backbone',
    'path': 'path/to/pretrained/weights',
    'type': object,
    'args': tuple(),
    'transform': resnet_transform,

    'selected_layers': list(),
    'pred_scales': list(),

    'use_pixel_scales': False,
    'preapply_sqrt': True,
    'use_square_anchors': False,
})

resnet101_backbone = backbone_base.copy({
    'name': 'ResNet101',
    'path': 'resnet101_reducedfc.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 23, 3],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]]*6,
})

resnet101_gn_backbone = backbone_base.copy({
    'name': 'ResNet101_GN',
    'path': 'R-101-GN.pkl',
    'type': ResNetBackboneGN,
    'args': ([3, 4, 23, 3],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]]*6,
})

resnet101_dcn_inter3_backbone = resnet101_backbone.copy({
    'name': 'ResNet101_DCN_Interval3',
    'args': ([3, 4, 23, 3], [0, 4, 23, 3], 3),
})

resnet50_backbone = resnet101_backbone.copy({
    'name': 'ResNet50',
    'path': 'resnet50-19c8e357.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 6, 3],),
    'transform': resnet_transform,
})

darknet53_backbone = backbone_base.copy({
    'name': 'DarkNet53',
    'path': 'darknet53.pth',
    'type': DarkNetBackbone,
    'args': ([1, 2, 8, 8, 4],),
    'transform': darknet_transform,

    'selected_layers': list(range(3, 9)),
    'pred_scales': [[3.5, 4.95], [3.6, 4.90], [3.3, 4.02], [2.7, 3.10], [2.1, 2.37], [1.8, 1.92]],
})

vgg16_arch = [[64, 64],
              [ 'M', 128, 128],
              [ 'M', 256, 256, 256],
              [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
              [ 'M', 512, 512, 512],
              [('M',  {'kernel_size': 3, 'stride':  1, 'padding':  1}),
               (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
               (1024, {'kernel_size': 1})]]

vgg16_backbone = backbone_base.copy({
    'name': 'VGG16',
    'path': 'vgg16_reducedfc.pth',
    'type': VGGBackbone,
    'args': (vgg16_arch, [(256, 2), (128, 2), (128, 1), (128, 1)], [3]),
    'transform': vgg_transform,

    'selected_layers': [3] + list(range(5, 10)),
    'pred_scales': [[5, 4]]*6,
})


# ----------------------- FPN DEFAULTS ----------------------- #

fpn_base = Config({
    # The number of features to have in each FPN layer
    'num_features': 256,

    # The upsampling mode used
    'interpolation_mode': 'bilinear',

    # The number of extra layers to be produced by downsampling starting at P5
    'num_downsample': 1,

    # Whether to down sample with a 3x3 stride 2 conv layer instead of just a stride 2 selection
    'use_conv_downsample': False,

    # Whether to pad the pred layers with 1 on each side (I forgot to add this at the start)
    # This is just here for backwards compatibility
    'pad': True,

    # Whether to add relu to the downsampled layers.
    'relu_downsample_layers': False,

    # Whether to add relu to the regular layers
    'relu_pred_layers': True,
})





# ----------------------- CONFIG DEFAULTS ----------------------- #

coco_base_config = Config({
    'dataset': coco2017_dataset,
    'num_classes': 81, # This should include the background class

    # The maximum number of detections for evaluation
    'max_num_detections': 100,

    # dw' = momentum * dw - lr * (grad + decay * w)
    'lr': 1e-3,
    'momentum': 0.9,
    'decay': 5e-4,

    # For each lr step, what to multiply the lr with
    'gamma': 0.1,
    'lr_steps': (280000, 360000, 400000),

    # Initial learning rate to linearly warmup from (if until > 0)
    'lr_warmup_init': 1e-4,

    # If > 0 then increase the lr linearly from warmup_init to lr each iter for until iters
    'lr_warmup_until': 2000,

    # The terms to scale the respective loss by
    'conf_alpha': 1,
    'mask_alpha': 0.4 / 256 * 140 * 140, # Some funky equation. Don't worry about it.

    # Eval.py sets this if you just want to run YOLACT as a detector
    'eval_mask_branch': True,

    # Top_k examples to consider for NMS
    'nms_top_k': 200,
    # Examples with confidence less than this are not considered by NMS
    'nms_conf_thresh': 0.05,
    # Boxes with IoU overlap greater than this threshold will be culled during NMS
    'nms_thresh': 0.5,

    'masks_to_train': 100,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})],
    'mask_proto_bias': False,
    'mask_proto_crop': True,

    # SSD data augmentation parameters
    # Randomize hue, vibrance, etc.
    'augment_photometric_distort': True,
    # Have a chance to scale down the image and pad (to emulate smaller detections)
    'augment_expand': False, #True,
    # Potentialy sample a random crop from the image and put it in a random place
    'augment_random_sample_crop': True, ##################################################
    # Mirror the image with a probability of 1/2
    'augment_random_mirror': True,
    # Flip the image vertically with a probability of 1/2
    'augment_random_flip': False,
    
    # Discard detections with width and height smaller than this (in absolute width and height)
    'discard_box_width': 4/550, #20 / 1280,
    'discard_box_height': 4/550, #20 / 720,

    # If using batchnorm anywhere in the backbone, freeze the batchnorm layer during training.
    # Note: any additional batch norm layers after the backbone will not be frozen.
    'freeze_bn': False,

    # Set this to a config object if you want an FPN (inherit from fpn_base). See fpn_base for details.
    'fpn': fpn_base.copy({
        'use_conv_downsample': True,
        'num_downsample': 2,
    }),

    # Use the same weights for each network head
    'share_prediction_module': True,

    # Use focal loss as described in https://arxiv.org/pdf/1708.02002.pdf instead of OHEM
    'use_focal_loss': False,
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2,
    
    # The initial bias toward forground objects, as specified in the focal loss paper
    'focal_loss_init_pi': 0.01,

    # Adds a 1x1 convolution directly to the biggest selected layer that predicts a semantic segmentations for each of the 80 classes.
    # This branch is only evaluated during training time and is just there for multitask learning.
    'use_semantic_segmentation_loss': False,
    'semantic_segmentation_alpha': 1,

    # Uses the same network format as mask_proto_net, except this time it's for adding extra head layers before the final
    # prediction in prediction modules. If this is none, no extra layers will be added.
    'extra_head_net': [(256, 3, {'padding': 1})],

    # What params should the final head layers have (the ones that predict box, confidence, and mask coeffs)
    'head_layer_params': {'kernel_size': 3, 'padding': 1},

    # Add extra layers between the backbone and the network heads
    # The order is (conf & centerness, polygon, mask)
    'extra_layers': (0, 0, 0),

    # When using ohem, the ratio between positives and negatives (3 means 3 negatives to 1 positive)
    'ohem_negpos_ratio': 3,

    # This is filled in at runtime by Yolact's __init__, so don't touch it
    'mask_dim': None,

    # Input image size.
    'max_size': 300,

    'train_masks': True,
    'train_boxes': True,
    'train_polygons': False,
    'train_centerness': False,

    'backbone': None,
    'name': 'base_config',
   
})


# ----------------------- CONVEXMASK CONFIGS ----------------------- #

convexmask_base_config = coco_base_config.copy({
    'name' : 'convexmask_base',
    
    # BACKBONE
    'backbone': resnet50_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': False,
        'pred_scales': [[24], [48], [96], [192], [384]],
        'strides': [8, 16, 32, 64, 128],
        'regress_ranges': ((-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)),
    }),

    'freeze_backbone':False,
    'freeze_blocks':[],

    # HEAD
    'extra_head_net': [(256, 3, {'padding': 1})],
    'num_rays' : 36,
    'ratio_distances':math.sqrt(2)/2,
    
    # ENCODING
    'force_gt_attribute':True,
    'radius':1.0,
    'inside_polygon':True,
    'regress_factor':1,

    'polar_centerness':False,
    'circular_centerness':True,

    # LOSSES
    'train_polygons': True,
    'train_centerness': True,
    'train_masks': True, 

    'use_focal_loss': True,
    'use_semantic_segmentation_loss':False,

    'mask_crop_convex': True,
    
    # COEFFICIENTS
    'conf_alpha': 1,
    'poly_alpha': 2, 
    'center_alpha': 1,
    'mask_alpha': 1, 
    'semantic_segmentation_alpha': 0,

    # POST PROCESSING
    'nms_poly': True,
    'extend_factor':0.10,
    'centerness_factor':0,

    # DATA AUGMENTATION
    'augment_photometric_distort': False, 
    'augment_expand': False,
    'augment_random_sample_crop': False, 
    'augment_random_mirror': True,
    'augment_random_flip': False,
    'albu_augment': False,
    'augment_resize':False,
    'resize_scales':[],
    
    # HYPERPARAMETERS
    'epochs':36,
    'batch_size':8,
    'lr_steps':(27,33),
    'lr':1e-2,
    'decay':1e-4,
    
    # OTHER STUFF
    'init_weights_folder': '/home/2017018/rconda01/weights/',

})

convex_coco_focal = convexmask_base_config.copy({
    'name':'convexmask_coco_600_ohem',

    # DATABASE
    'dataset': coco2017_dataset,
    'num_classes': len(coco2017_dataset.class_names) + 1,
    'max_size': (1330, 800),
    'fixed_size':False,

    'nms_poly':False,
    'extend_factor':0.10,

    'albu_augment':False,
    'augment_photometric_distort': False, 
    'augment_expand': False, 
    'augment_random_sample_crop': False, 
    'augment_random_mirror': True,
    'augment_resize': False,
    'resize_scales':COCO_RESIZE_SCALES[800],

})

convex_coco_focal_101 = convex_coco_focal.copy({
    'name':'COCO_800_R101',

    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': False,
        'pred_scales': [[24], [48], [96], [192], [384]],
        'strides': [8, 16, 32, 64, 128],
        'regress_ranges': ((-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)),
    })

})

convex_synthtree_focal = convex_coco_focal.copy({
    'name':'ABLATION_NR_72_k2',
    'dataset': synthtree43k_dataset,
    'num_classes': 2,
    'max_size': (1280,720),
    'fixed_size':True,
    
    'epochs':29, #AUCUNE DATA AUG DONC CA OVERFIT RAPIDE

    'polar_centerness':False,
    'circular_centerness':True,

    'num_rays':72,

    'nms_poly':True,
    'nms_thresh':0.5,

    'albu_augment':False,
    'augment_resize': False,
    'augment_photometric_distort': False,
    'augment_expand': False,
    'augment_random_sample_crop': False,
    'augment_random_mirror': True,

})


convex_synthtree_focal_101 = convex_synthtree_focal.copy({
    'name':'ABLATION_NR_72_R101',
    'num_rays':72,
    'centerness_factor':0,
    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': False,
        'pred_scales': [[24], [48], [96], [192], [384]],
        'strides': [8, 16, 32, 64, 128],
        'regress_ranges': ((-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)),
    })

})


#-----------------------------------------------------



convex_synthtree_prediction = convex_synthtree_focal.copy({
    'name':'tmp', 
     
    'freeze_bn':True, #False,
    'batch_size':4,

    'lr':1e-2,
    'decay':1e-4,

    'epochs':36,
    'num_rays':72,

    'freeze_backbone':False, 
    'freeze_blocks':[-1,0], #0 -> 3 : ResNet blocks ; -1 : ResNet stem


    'albu_augment':True,
    'augment_photometric_distort':False, 
    'augment_resize':True, 
    'augment_random_mirror':False,

    'resize_scales':SYNTH_RESIZE_SCALES[720],
})


convex_synthtree_prediction_101 = convex_synthtree_prediction.copy({
    'name':'PREDICTION_101_bs8_freeze_full_6x_criann',

    'freeze_backbone': True,
    'freeze_blocks':[-1,0,1,2,3],

    'batch_size':8,
    'freeze_bn':True,

    'epochs':72, #36,
    'lr_steps':(54,66), #(27,33),

    'centerness_factor':0,
    'nms_thresh':0.5,
    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': False,
        'pred_scales': [[24], [48], [96], [192], [384]],
        'strides': [8, 16, 32, 64, 128],
        'regress_ranges': ((-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)),
    })

    
})


# Default config
cfg = convex_coco_focal.copy()

def set_cfg(config_name:str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))

    if cfg.name is None:
        cfg.name = config_name.split('_config')[0]

def set_dataset(dataset_name:str):
    """ Sets the dataset of the current config. """
    cfg.dataset = eval(dataset_name)
    
