# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
#                                    OmniPose                                    #
#      Rochester Institute of Technology - Vision and Image Processing Lab       #
#                      Bruno Artacho (bmartacho@mail.rit.edu)                    #
# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #

from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
import pprint
# import _init_paths
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..'))
import warnings

import cython
from tqdm import tqdm

import dataset
import models
from config import cfg, update_config
from core.function import validate
from core.loss import JointsMSELoss
from models.omnipose import get_omnipose
from models.pose_hrnet import get_pose_net
from utils.utils import create_logger

warnings.filterwarnings("ignore") 



import argparse
import os
import pdb
import pprint
# import _init_paths
import sys

import cv2
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..'))
import cython

import dataset
import models
from config import cfg, update_config
from core.function import validate
# from models.omnipose import get_Canny_HRNet
# from models.frankenstein import get_frankenstein
from core.inference import get_final_preds_no_transform
from core.loss import JointsMSELoss
from models.omnipose import OmniPose, get_omnipose
from utils.utils import create_logger


class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i])/255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i])/255.))

        # Red    = (240,  2,127)
        # Yellow = (255,255,  0)
        # Green  = (169,209,142)
        # Pink   = (252,176,243)
        # Blue   = (0,176,240)
        color_ids = [(0,176,240), (252,176,243), (169,209,142), (255,255,  0), (240,2,127)]

        self.color_ids = []
        for i in range(len(color_ids)):
            self.color_ids.append(tuple(np.array(color_ids[i])/255.))


# color = [(252,176,243),(252,176,243),(252,176,243),
#             (0,176,240), (0,176,240), (0,176,240),
#             (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127), 
#             (255,255,0), (255,255,0),(169, 209, 142),
#             (169, 209, 142),(169, 209, 142)]
# link_pairs = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], \
#     [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], \
#     [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]]

color = [[000,000,255], [000,255,000], [000,000,255], [255,255,000], [255,255,000], [255,000,255], [000,255,000],\
              [255,000,000], [255,255,000], [255,000,255], [000,255,000], [000,255,000], [000,000,255], [255,255,000], [255,000,000]]
link_pairs = [[ 8, 9], [ 7,12], [12,11], [11,10], [ 7,13], [13,14], [14,15], [ 7, 6], [ 6, 2], [ 2, 1], [ 1, 0], [ 6, 3], [ 3, 4], [ 4, 5], [ 7, 8]]


point_color = [(240,2,127),(240,2,127),(240,2,127), 
            (240,2,127), (240,2,127), 
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (252,176,243),(0,176,240),(252,176,243),
            (0,176,240),(252,176,243),(0,176,240),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142)]

artacho_style = ColorStyle(color, link_pairs, point_color)

def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)
        
    return joints_dict

def plot_COCO_image(preds, img_path, save_path, link_pairs, ring_color, color_ids, save=True):
    # Read Images
    data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    data_numpy = cv2.resize(data_numpy, (384,288), interpolation = cv2.INTER_AREA)
    h = data_numpy.shape[0]
    w = data_numpy.shape[1]
    
    # Plot
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = plt.subplot(1,1,1)
    bk = plt.imshow(data_numpy[:,:,::-1])
    bk.set_zorder(-1)
    joints_dict = map_joint_dict(preds[0])
    
    # stick 
    for k, link_pair in enumerate(link_pairs):
        lw = 2
        line = mlines.Line2D(
                np.array([joints_dict[link_pair[0]][0],
                          joints_dict[link_pair[1]][0]]),
                np.array([joints_dict[link_pair[0]][1],
                          joints_dict[link_pair[1]][1]]),
                ls='-', lw=lw, alpha=1, color=color_ids[0],)
        line.set_zorder(0)
        ax.add_line(line)
    # black ring
    for k in range(preds.shape[1]):
        if preds[0,k,0] > w or preds[0,k,1] > h:
            continue
        radius = 2

        circle = mpatches.Circle(tuple(preds[0,k,:2]), 
                                 radius=radius, 
                                 ec='black', 
                                 fc=ring_color[k], 
                                 alpha=1, 
                                 linewidth=1)
        circle.set_zorder(1)
        ax.add_patch(circle)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)        
    plt.margins(0,0)
    print(save_path)
    plt.savefig(save_path, format='jpg', bbox_inckes='tight', dpi=100)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',          help='experiment configure file name',
                        default='experiments/mpii/hrnet/w48_256x256_adam_lr1e-3.yaml', type=str)
    parser.add_argument('--opts',         help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir',     help='model directory', type=str, default='')
    parser.add_argument('--logDir',       help='log directory', type=str, default='')
    parser.add_argument('--dataDir',      help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')

    args = parser.parse_args()
    return args

def main(args):
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.MODEL.NAME == 'pose_hrnet':
        model = get_pose_net(cfg, is_train=True)
    elif cfg.MODEL.NAME == 'omnipose':
        model = get_omnipose(cfg, is_train=True)

    if cfg.TEST.MODEL_FILE:
        logger.info("=> loading checkpoint '{}'".format(cfg.TEST.MODEL_FILE))
        checkpoint = torch.load(cfg.TEST.MODEL_FILE)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        
        print('Loading checkpoint with accuracy of ', checkpoint['perf'], 'at epoch ',checkpoint['epoch'])

        model_state_dict = model.state_dict()
        new_model_state_dict = {}
        
        for k in checkpoint['state_dict']:
            # print(k)
            if k in model_state_dict and model_state_dict[k].size() == checkpoint['state_dict'][k].size():
                new_model_state_dict[k] = checkpoint['state_dict'][k]
            else:
                print('Skipped loading parameter {}'.format(k))

        model.load_state_dict(checkpoint, strict=False)

        print('begin_epoch', begin_epoch)
        print('best_perf', best_perf)
        print('last_epoch',last_epoch)

        model.load_state_dict(new_model_state_dict, strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        model_state_file = './output/mpii/omnipose/omnipose_w48_256x256/checkpoint.pth'
        logger.info('=> loading model from {}'.format(model_state_file))

        model_state_dict = torch.load(model_state_file)['best_state_dict']
        new_model_state_dict = {}
        for k in model_state_dict:
            if k in model_state_dict: #  and model_state_dict[k].size() == model_state_dict[k].size():
                new_model_state_dict[k] = model_state_dict[k]
            else:
                print('Skipped loading parameter {}'.format(k))

        model.load_state_dict(new_model_state_dict, strict=False)

    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,]))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True)

    # evaluate on validation set
    # validate(cfg, valid_loader, valid_dataset, cfg.DATASET.DATASET, model, criterion, final_output_dir, tb_log_dir)

    # print('final_output_dir: ',final_output_dir)

    model.eval()
    with torch.no_grad():
        tbar = tqdm(valid_loader)
        for i, (input, target, target_weight, meta) in enumerate(tbar):
            input  = input.cuda()
            outputs = model(input)

            preds, maxvals = get_final_preds_no_transform(cfg, outputs.detach().cpu().numpy())

            colorstyle = artacho_style
            plot_COCO_image(4*preds, img_path, 'samples/mpii/'+images[idx], colorstyle.link_pairs, colorstyle.ring_color, colorstyle.color_ids, save=True)

            pdb.set_trace()



if __name__ == '__main__':
    arg = parse_args()
    main(arg)
