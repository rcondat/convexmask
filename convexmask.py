import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
import numpy as np
from itertools import product
from math import sqrt
from typing import List
from collections import defaultdict
import math
from data.config import cfg
from layers import Detect
from layers.interpolate import InterpolateModule
from backbone import construct_backbone

import torch.backends.cudnn as cudnn
from utils import timer
from utils.functions import MovingAverage, make_net
import os
import numpy as np

from data.config import MEANS, STD
import cv2

torch.cuda.current_device()
use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn


prior_cache = defaultdict(lambda: None)

class PredictionModule(nn.Module):
    """
    ConvexMask prediction module. 
    Contrary to traditionnal object detectors, ConvexMask predicts a convex exterior polygon instead of a bounding box.
    
    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """
    
    def __init__(self, in_channels, out_channels=1024, scales=[1], stride=1, parent=None, index=0, regress_ranges = (0,0)):
        super().__init__()
        
        self.num_classes = cfg.num_classes
        self.mask_dim    = cfg.mask_dim # Defined by Yolact
        self.parent      = [parent] # Don't include this in the state dict
        self.index       = index
        self.num_heads   = cfg.num_heads # Defined by Yolact
        self.stride      = stride
        self.num_rays    = cfg.num_rays
        self.regress_ranges = [i*cfg.regress_factor for i in regress_ranges] 
        self.ratio_distances = cfg.ratio_distances
        
        if parent is None:
            if cfg.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)

            self.conf_layer = nn.Conv2d(out_channels, self.num_classes, **cfg.head_layer_params) # Classification
            self.center_layer = nn.Conv2d(out_channels, 1, **cfg.head_layer_params) # Centerness
            self.polygon_layer = nn.Conv2d(out_channels, self.num_rays, **cfg.head_layer_params) # Polygon
            self.mask_layer = nn.Conv2d(out_channels, self.mask_dim,    **cfg.head_layer_params) # Mask Segmentation
            
            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.conf_extra, self.poly_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]
            
        self.scales = scales

        self.priors = None
        self.last_conv_size = None
        self.last_img_size = None

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (class_confs, centerness, polygons_rays, mask_output, prior_boxes) with sizes
            - class_confs: [batch_size, conv_h*conv_w, num_classes]
            - centerness: [batch_size, conv_h*conv_w, 1]
            - polygons_rays: [batch_size, conv_h*conv_w, num_rays]
            - mask_output: [batch_size, conv_h*conv_w, mask_dim]
            - prior_boxes: [conv_h*conv_w, 4]
        """
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]
        
        conv_h = x.size(2)
        conv_w = x.size(3)
        
        if cfg.extra_head_net is not None:
            x = src.upfeature(x)

        conf_x = src.conf_extra(x)
        poly_x = src.poly_extra(x)
        mask_x = src.mask_extra(x)

        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        center = src.center_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)

        #ratio_distances = min(math.sqrt((self.regress_ranges[1]/cfg._tmp_img_h)**2+(self.regress_ranges[1]/cfg._tmp_img_w)**2),math.sqrt(2))
        ratio_distances = min(math.sqrt((self.regress_ranges[1]/cfg.max_size[0])**2+(self.regress_ranges[1]/cfg.max_size[0])**2),math.sqrt(2))
        polygon = src.polygon_layer(poly_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_rays).sigmoid() * ratio_distances
        
        mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)



        

        if cfg.eval_mask_branch:
            mask = torch.tanh(mask)
        
        self.last_img_size = (cfg._tmp_img_w, cfg._tmp_img_h)
        self.last_conv_size = (conv_w, conv_h)
        points = self.get_points_single((conv_w,conv_h),self.stride, torch.float, conf.device)
        regress_ranges = points.new_tensor(self.regress_ranges)[None].expand_as(points)
        
        if cfg.center_alpha==0:
            center = torch.ones_like(center)

        preds = {'conf': conf, 'center': center, 'polygon': polygon, 'mask': mask, 'points': points, 
                 'regress_ranges': regress_ranges} 



        
        return preds


    def get_points_single(self, featmap_size, stride, dtype, device):
        w,h = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)  
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device) 
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        img_size = torch.Tensor(self.last_img_size)
        points /= img_size
        return points

    
class FPN(ScriptModuleWrapper):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Parameters (in cfg.fpn):
        - num_features (int): The number of output features in the fpn layers.
        - interpolation_mode (str): The mode to pass to F.interpolate.
        - num_downsample (int): The number of downsampled layers to add onto the selected layers.
                                These extra layers are downsampled from the last selected layer.

    Args:
        - in_channels (list): For each conv layer you supply in the forward pass,
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample', 'relu_pred_layers',
                     'lat_layers', 'pred_layers', 'downsample_layers', 'relu_downsample_layers']

    def __init__(self, in_channels):
        super().__init__()

        self.lat_layers  = nn.ModuleList([
            nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1)
            for x in reversed(in_channels)
        ])

        # This is here for backwards compatability
        padding = 1 if cfg.fpn.pad else 0
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        if cfg.fpn.use_conv_downsample:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2)
                for _ in range(cfg.fpn.num_downsample)
            ])
        
        self.interpolation_mode     = cfg.fpn.interpolation_mode
        self.num_downsample         = cfg.fpn.num_downsample
        self.use_conv_downsample    = cfg.fpn.use_conv_downsample
        self.relu_downsample_layers = cfg.fpn.relu_downsample_layers
        self.relu_pred_layers       = cfg.fpn.relu_pred_layers

    @script_method_wrapper
    def forward(self, convouts:List[torch.Tensor]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            
            x = x + lat_layer(convouts[j])
            out[j] = x
        
        # This janky second loop is here because TorchScript.
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = pred_layer(out[j])

            if self.relu_pred_layers:
                F.relu(out[j], inplace=True)

        cur_idx = len(out)

        # In the original paper, this takes care of P6
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        if self.relu_downsample_layers:
            for idx in range(len(out) - cur_idx):
                out[idx] = F.relu(out[idx + cur_idx], inplace=False)

        return out



class ConvexMask(nn.Module):
    """

    You can set the arguments by changing them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self):
        super().__init__()


        train_vars = [True,cfg.train_polygons,cfg.train_masks,cfg.train_centerness,cfg.use_semantic_segmentation_loss]
        self.loss_weights = nn.Parameter(torch.zeros((sum(train_vars))))

        self.backbone = construct_backbone(cfg.backbone)

        if cfg.freeze_bn:
            self.freeze_bn()

        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!

        self.num_grids = 0

        self.proto_src = 0
        
        if self.proto_src is None: in_channels = 3
        elif cfg.fpn is not None: in_channels = cfg.fpn.num_features
        else: in_channels = self.backbone.channels[self.proto_src]
        in_channels += self.num_grids

        # The include_last_relu=false here is because we might want to change it to another function
        self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)

        if cfg.mask_proto_bias:
            cfg.mask_dim += 1


        self.selected_layers = cfg.backbone.selected_layers
        src_channels = self.backbone.channels

        if cfg.fpn is not None:
            # Some hacky rewiring to accomodate the FPN
            self.fpn = FPN([src_channels[i] for i in self.selected_layers])
            self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)


        self.prediction_layers = nn.ModuleList()
        cfg.num_heads = len(self.selected_layers)

        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]

            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                    scales        = cfg.backbone.pred_scales[idx],
                                    stride        = cfg.backbone.strides[idx],
                                    parent        = parent,
                                    index         = idx,
                                    regress_ranges = cfg.backbone.regress_ranges[idx])
            self.prediction_layers.append(pred)

        # Extra parameters for the extra losses       
        if cfg.use_semantic_segmentation_loss:
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg.num_classes-1, kernel_size=1)

        # For use in evaluation
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k,
            conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)


    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def save_checkpoint(self, path, epoch, best_loss, optimizer, scaler):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save({'model':self.state_dict(),
                    'epoch':epoch,
                    'best_valid_loss':best_loss,
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict()
                    }, path)
   

    def load_checkpoint(self, path, optimizer, scaler):
        state_dict = torch.load(path)
        start_epoch = state_dict['epoch']
        best_loss = state_dict['best_valid_loss']
        optimizer.load_state_dict(state_dict['optimizer'])
        scaler.load_state_dict(state_dict['scaler'])

        state_dict = state_dict['model']
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]

            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]
        self.load_state_dict(state_dict)
        return optimizer, scaler, start_epoch, best_loss       

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]
        
            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]
            
            if 'conf_layer' in key or 'polygon_layer':
                for name, para in self.named_parameters():
                    if name==key:
                        if para.shape != state_dict[key].shape:
                            del state_dict[key]
        self.load_state_dict(state_dict,strict=False)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')
        
        # Quick lambda to test if one list contains the other
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
            # Broke in 1.4 (see issue #292), where RecursiveScriptModule is the new star of the show.
            # Note that this might break with future pytorch updates, so let me know if it does
            is_script_conv = False
            if 'Script' in type(module).__name__:
                # 1.4 workaround: now there's an original_name member so just use that
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                # 1.3 workaround: check if this has the same constants as a conv module
                else:
                    is_script_conv = (
                        all_in(module.__dict__['_constants_set'], conv_constants)
                        and all_in(conv_constants, module.__dict__['_constants_set']))
            
            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv

            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)
                
                if module.bias is not None:
                    if cfg.use_focal_loss and 'conf_layer' in name:
                        module.bias.data[0]  = -np.log(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi))
                        module.bias.data[1:] = -np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()
            """
            if 'center_layer' in name:
                module.weight.data.zero_()
                nn.init.constant_(module.bias.data,-6)
            """
    def train(self, mode=True):
        super().train(mode)

        if cfg.freeze_bn:
            self.freeze_bn()

        if cfg.freeze_backbone:
            self.freeze_backbone(blocks=cfg.freeze_blocks)

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable
    
    def freeze_backbone(self, enable=False, blocks=[0,1,2,3]):
        
        block_keys = ['backbone.layers.{}'.format(b) for b in blocks]
        for name,para in self.named_parameters():
            if np.sum([b in name for b in block_keys])>0:
                para.requires_grad = enable
            if 'backbone' in name and 'layers' not in name:
                # Special case for the first layers
                if -1 in blocks:
                    para.requires_grad = enable
        


    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        _, _, img_h, img_w = x.size()

        cfg._tmp_img_h = img_h
        cfg._tmp_img_w = img_w
        
        with timer.env('backbone'):
            outs = self.backbone(x)

        if cfg.fpn is not None:
            with timer.env('fpn'):
                # Use backbone.selected_layers because we overwrote self.selected_layers
                outs = [outs[i] for i in cfg.backbone.selected_layers]
                outs = self.fpn(outs)

        proto_out = None
        if cfg.eval_mask_branch:
            with timer.env('proto'):
                proto_x = x if self.proto_src is None else outs[self.proto_src]
                
                if self.num_grids > 0:
                    grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
                    proto_x = torch.cat([proto_x, grids], dim=1)

                proto_out = self.proto_net(proto_x)
                proto_out = torch.relu(proto_out)
                
                # Move the features last so the multiplication is easy
                proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

                if cfg.mask_proto_bias:
                    bias_shape = [x for x in proto_out.size()]
                    bias_shape[-1] = 1
                    proto_out = torch.cat([proto_out, torch.ones(*bias_shape)], -1)


        with timer.env('pred_heads'):
            pred_outs = {'conf': [], 'center': [], 'polygon': [], 'mask': [], 'points': [], 'regress_ranges': []} 
            
            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
                pred_x = outs[idx]

                # A hack for the way dataparallel works
                if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]:
                    pred_layer.parent = [self.prediction_layers[0]]

                p = pred_layer(pred_x)
                
                for k, v in p.items():
                    pred_outs[k].append(v)

        for k, v in pred_outs.items():
            if type(pred_outs[k][0]) != int:
                pred_outs[k] = torch.cat(v, -2)

        if proto_out is not None:
            pred_outs['proto'] = proto_out

        pred_outs['img_size'] = (img_h,img_w)


        if self.training:
            # For the extra loss functions
            if cfg.use_semantic_segmentation_loss:
                pred_outs['segm'] = self.semantic_seg_conv(outs[0])

            return pred_outs
        else:
            pred_outs['center']=torch.sigmoid(pred_outs['center'])

            if cfg.use_focal_loss:
                pred_outs['conf'] = torch.sigmoid(pred_outs['conf'])

            else:
                pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)

            return self.detect(pred_outs, self)

# Some testing code
if __name__ == '__main__':
    from utils.functions import init_console
    init_console()

    # Use the first argument to set the config if you want
    import sys
    if len(sys.argv) > 1:
        from data.config import set_cfg
        set_cfg(sys.argv[1])

    net = ConvexMask()
    net.train()
    net.init_weights(backbone_path=cfg.init_weights_folder + cfg.backbone.path)

    # GPU
    net = net.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    x = torch.rand((1, 3, cfg.max_size[0], cfg.max_size[0]))
    y = net(x)

    for p in net.prediction_layers:
        print(p.last_conv_size)

    print()
    for k, a in y.items():
        print(k + ': ', a.size(), torch.sum(a))
    
    net(x)
    # timer.disable('pass2')
    avg = MovingAverage()
    try:
        while True:
            timer.reset()
            with timer.env('everything else'):
                net(x)
            avg.add(timer.total_time())
            print('\033[2J') # Moves console cursor to 0,0
            timer.print_stats()
            print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))
    except KeyboardInterrupt:
        pass
