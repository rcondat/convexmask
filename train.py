from data import *

from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from convexmask import ConvexMask
import os
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime

from tensorboard import SummaryWriter

from torch.cuda.amp import GradScaler, autocast

# Oof
import eval as eval_script

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description='Yolact Training Script')

parser.add_argument('--home_dir',default='/home/2017018/rconda01/fold_results/', type=str)
parser.add_argument('--tb_dir',default='tensorboard/', type=str)
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--init_weights_folder',default='../weights/')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models.')
parser.add_argument('--log_folder', default='logs/',
                    help='Directory for saving logs.')
parser.add_argument('--config', default=None,
                    help='The config object to use.')
parser.add_argument('--validation_size', default=5000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--log_iter', default=50,type=int)
parser.add_argument('--validation_epoch', default=2, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')
parser.add_argument('--trained_model',
                    default=None, type=str,
                    help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')


parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

amp=True

os.makedirs(args.home_dir,exist_ok=True)

args.log_folder = args.home_dir+args.log_folder
if args.config is not None:
    set_cfg(args.config)

if args.dataset is not None:
    set_dataset(args.dataset)

if args.autoscale and cfg.batch_size != 8:
    factor = cfg.batch_size / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, cfg.batch_size))

    cfg.lr *= factor
    #cfg.max_iter //= factor
    #cfg.lr_steps = [x // factor for x in cfg.lr_steps]

# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

if args.kfold is not None:
   cfg.dataset.train_info = cfg.dataset.train_info.replace('fold_00','fold_0{}'.format(args.kfold))
   cfg.dataset.valid_info = cfg.dataset.valid_info.replace('fold_00','fold_0{}'.format(args.kfold))
   print(cfg.dataset.train_info)
   print(cfg.dataset.valid_info)
   cfg.name = cfg.name+'_k{}'.format(args.kfold)

# This is managed by set_lr
cur_lr = args.lr

args.save_folder = args.home_dir+args.save_folder+cfg.name+'/'
args.tb_dir = args.home_dir+args.tb_dir+cfg.name+'/'
os.makedirs(args.tb_dir,exist_ok=True)
os.makedirs(args.save_folder,exist_ok=True)
if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if cfg.batch_size // torch.cuda.device_count() < 6:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['C', 'M', 'P', 'N', 'D', 'E', 'K', 'S', 'I', 'X']

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """
    
    def __init__(self, net:ConvexMask, criterion:MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion
    
    def forward(self, images, targets, masks, polygons, num_crowds):
        preds = self.net(images)
        """
        path = '/home/2017018/rconda01/visu/'
        os.makedirs(path,exist_ok=True)
        os.makedirs(path+'images/',exist_ok=True)
        bs = images.shape[0]
        for i in range(bs):
            img = images[i]
            t = img.moveaxis(0,2).cpu().numpy()[...,[2,1,0]]
            mean = np.tile(MEANS, (t.shape[0], t.shape[1], 1)).astype(np.float32)
            std = np.tile(STD, (t.shape[0], t.shape[1], 1)).astype(np.float32)
            t = ((t * std)+mean).astype(np.uint8)
            cv2.imwrite(path+'images/{}.png'.format(i),t)

        os.makedirs(path+'masks/',exist_ok=True)
        for i,mask in enumerate(masks):
            for j in range(mask.shape[0]):
                t = mask[j].cpu().numpy()
                cv2.imwrite(path+'masks/{}_{}.png'.format(i,j),t)
        
        os.makedirs(path+'boxes/',exist_ok=True)
        for i, box in enumerate(targets):
            np.savetxt(path+'boxes/{}.txt'.format(i),box.cpu().numpy())
        
        os.makedirs(path+'polygons/',exist_ok=True)
        for i, poly in enumerate(polygons):
            for j in range(poly.shape[0]):
                np.savetxt(path+'polygons/{}_{}.txt'.format(i,j),poly[j].cpu().numpy())

        exit()
        """
        losses = self.criterion(self.net, preds, targets, masks, polygons, num_crowds)
        return losses

class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])
        
        return out

def train():
    print("START")
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS))
    
    val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                info_file=cfg.dataset.valid_info,
                                transform=BaseTransform(MEANS,resize_gt=True)) ####


    if args.validation_epoch > 0:
        setup_eval()
        test_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS,resize_gt=False))

    print("DATASET CREATED")

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = ConvexMask()
    net = yolact_net
    net.train()

    print("POLAR YOLACT CREATED")

    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),
            overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)
        tb_log = SummaryWriter(log_dir=args.tb_dir) 

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)

    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                                  num_rays=cfg.num_rays,
                                  negpos_ratio=cfg.ohem_negpos_ratio)

    print("POLAR MULTIBOX LOSS CREATED")

    scaler = GradScaler(enabled=amp)

    if os.path.exists(args.save_folder+'/checkpoint.pth'):
        print('Resuming training, loading {}/checkpoint.pth'.format(args.save_folder))
        optimizer, scaler, start_epoch, best_val_loss = yolact_net.load_checkpoint('{}/checkpoint.pth'.format(args.save_folder),optimizer,scaler)
    else:
        print('Initializing weights...')
        print(cfg.init_weights_folder + cfg.backbone.path)
        yolact_net.init_weights(backbone_path=cfg.init_weights_folder + cfg.backbone.path)
        best_val_loss = 10000
        start_epoch = 0
        if args.trained_model is not None:
            print('Load weights from {}'.format(args.trained_model))
            yolact_net.load_weights(args.trained_model)

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != cfg.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, cfg.batch_size))
            exit(-1)

    net = CustomDataParallel(NetLoss(net, criterion))
    if args.cuda:
        net = net.cuda()
    

    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size[1], cfg.max_size[0]).cuda())
    if not cfg.freeze_bn: 
        yolact_net.freeze_bn(True)
        yolact_net.train()

    epoch_size = len(dataset) // cfg.batch_size

    # loss counters
    last_time = time.time()
    iteration = epoch_size * (start_epoch)

    #num_epochs = math.ceil(cfg.max_iter / epoch_size)
    num_epochs = cfg.epochs
    num_iterations = cfg.epochs * epoch_size
    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    data_loader = data.DataLoader(dataset, cfg.batch_size,
                                  num_workers=8, drop_last=True,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True, generator=torch.Generator(device='cuda'))
    
    val_loader = data.DataLoader(val_dataset, cfg.batch_size,
                                 num_workers=8, drop_last=True,
                                 shuffle=False, collate_fn=detection_collate,
                                 pin_memory=True, generator=torch.Generator(device='cuda'))

    time_avg = MovingAverage()
    
    print("DATALOADERS CREATED")

    global loss_types # Forms the print order
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }
    print('Begin training!')
    print()
    
    # try-except so you can use ctrl+c to save early and stop training
    for epoch in range(start_epoch,num_epochs):
        all_losses = {k : [] for k in loss_types}
        val_losses = {k : [] for k in loss_types}

        while step_index < len(cfg.lr_steps) and epoch >= cfg.lr_steps[step_index]:
            step_index += 1
            set_lr(optimizer, args.lr * (args.gamma ** step_index))
        #'''        
        for datum in data_loader:

            # Warm up by linearly interpolating the learning rate from some smaller value
            if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)
            """
            # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
            while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                step_index += 1
                set_lr(optimizer, args.lr * (args.gamma ** step_index))
            """ 
            # Zero the grad to get ready to compute gradients
            optimizer.zero_grad()

            # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
            with autocast(enabled=amp): ###
                losses = net(datum)
                
            losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
            loss = sum([losses[k] for k in losses])
            

            # Backprop
            scaler.scale(loss).backward() ###

            scaler.step(optimizer) ###
            scaler.update() ###
            # Add the loss to the moving average for bookkeeping
            for k in losses:
                loss_avgs[k].add(losses[k].item())
                all_losses[k].append(losses[k].item())

            cur_time  = time.time()
            elapsed   = cur_time - last_time
            last_time = cur_time

            # Exclude graph setup from the timing information
            time_avg.add(elapsed)

            if iteration % args.log_iter == 0:
                eta_str = str(datetime.timedelta(seconds=(num_iterations-iteration) * time_avg.get_avg())).split('.')[0]
                    
                total = sum([loss_avgs[k].get_avg() for k in losses])
                loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                    
                print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                        % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

            if args.log:
                precision = 5
                loss_info = {k: round(losses[k].item(), precision) for k in losses}
                loss_info['T'] = round(loss.item(), precision)

                if args.log_gpu:
                    log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow
                        
                log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                    lr=round(cur_lr, 10), elapsed=elapsed)

                log.log_gpu_stats = args.log_gpu
                    
            iteration += 1

        mean_train_losses = {}
        for k,v in all_losses.items():
            if len(v)!=0:
                mean_train_losses[k] = np.mean(v)
        mean_train_losses['T'] =  sum([mean_train_losses[k] for k in losses])

        # TB LOG
        for k,v in mean_train_losses.items():
            tb_log.add_scalar('train/'+k,np.mean(v), epoch)
        #'''            

        print("Validation step")
           
        # Validation
        for datum in val_loader:
            
            optimizer.zero_grad()
            with autocast(enabled=amp):
                with torch.no_grad():
                    losses = net(datum)

            losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
            loss = sum([losses[k] for k in losses])

            for k in losses:
                val_losses[k].append(losses[k].item())

        mean_val_losses = {}
        for k,v in val_losses.items():
            if len(v) !=0:
               mean_val_losses[k] = np.mean(v)
        mean_val_losses['T'] = sum([mean_val_losses[k] for k in mean_val_losses])
        for k,v in mean_val_losses.items():
            print('{}: {:.3f}'.format(k,v))
            tb_log.add_scalar('val/'+k,v, epoch)
        
        # This is done per epoch
        if args.validation_epoch > 0:
            if epoch % args.validation_epoch == 0:
                compute_validation_map(epoch, iteration, yolact_net, test_dataset, log if args.log else None)
        """
        print("YOLACT REQUIRES GRAD")
        for name,para in yolact_net.named_parameters():
            try:
                print("{} : {}".format(name,para.requires_grad))
            except:
                print(name)
        exit()
        """

        yolact_net.save_checkpoint('{}/checkpoint.pth'.format(args.save_folder), epoch+1, best_val_loss, optimizer, scaler)
        
        # Save best model
        if mean_val_losses['T'] < best_val_loss:
            print("Best val loss : {} - Saving best model".format(mean_val_losses['T']))
            yolact_net.save_weights('{}/best_checkpoint.pth'.format(args.save_folder))
            best_val_loss = mean_val_losses['T']
        
    tb_log.close()

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    global cur_lr
    cur_lr = new_lr

def gradinator(x):
    x.requires_grad = False
    return x


def prepare_data(datum, devices:list=None, allocation:list=None):
    with torch.no_grad():
        images, (targets, masks, polygons, num_crowds), __, __ = datum

        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [len(images) // len(devices)] * (len(devices) - 1)
            allocation.append(len(images) - sum(allocation)) # The rest might need more/less

        cur_idx=0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx]   = gradinator(images[cur_idx].to(device))
                targets[cur_idx]  = gradinator(targets[cur_idx].to(device))
                masks[cur_idx]    = gradinator(masks[cur_idx].to(device))
                polygons[cur_idx] = gradinator(polygons[cur_idx].to(device))
                cur_idx += 1

        cur_idx = 0
        split_images, split_targets, split_masks, split_polygons, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(5)]
        for device_idx, alloc in enumerate(allocation):
            # IMAGES
            batch_images = images[cur_idx:cur_idx+alloc]
            batch_shapes = [i.shape for i in batch_images]
            if cfg.fixed_size:
                max_height = cfg.max_size[1]
                max_width = cfg.max_size[0]
            else:
                max_height = cfg.max_size[0]
                max_width = cfg.max_size[0]
            ratio_height = [b[1]/max_height for b in batch_shapes]
            ratio_width = [b[2]/max_width for b in batch_shapes]
            unified_images = batch_images[0].new_ones((len(batch_shapes),3,max_height,max_width))
            for i,shape,img in zip(range(len(batch_shapes)),batch_shapes,batch_images):
                unified_images[i,:,:shape[1],:shape[2]] = img
            split_images[device_idx]    = unified_images 
            # MASKS
            batch_masks = masks[cur_idx:cur_idx+alloc]
            unified_masks = []
            for mask in batch_masks:
                nb_mask,mask_height,mask_width = mask.shape
                base_mask = mask.new_zeros(nb_mask,max_height,max_width)
                base_mask[:,:mask_height,:mask_width] = mask
                unified_masks.append(base_mask)
            split_masks[device_idx]     = unified_masks #masks[cur_idx:cur_idx+alloc]

            # TARGETS
            batch_targets = targets[cur_idx:cur_idx+alloc]
            for tar, rat_h, rat_w in zip(batch_targets,ratio_height,ratio_width):
                tar[:,[0,2]]*=rat_w
                tar[:,[1,3]]*=rat_h
            split_targets[device_idx]   = batch_targets

            # POLYGONS
            batch_polygons = polygons[cur_idx:cur_idx+alloc]
            for poly, rat_h, rat_w in zip(batch_polygons,ratio_height,ratio_width):
                no_pts = poly==-1
                poly[...,0]*=rat_w
                poly[...,1]*=rat_h
                poly[no_pts] = -1
            split_polygons[device_idx]  = batch_polygons
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]
            cur_idx += alloc
        return split_images, split_targets, split_masks, split_polygons, split_numcrowds


def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None):
    with torch.no_grad():
        yolact_net.eval()
        
        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()

def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])

if __name__ == '__main__':
    train()
