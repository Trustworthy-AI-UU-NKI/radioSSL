from __future__ import print_function

import argparse
import os
import warnings
import sys
import time
import math
import numpy as np
import random
import seaborn as sns

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as f
from torch import autocast
if torch.cuda.is_available():
    from torch.cuda.amp import GradScaler
else:
    from torch.cpu.amp import GradScaler

from torch_kmeans import KMeans


warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Training benchmark')
    parser.add_argument('--data', metavar='DIR', default='/home/igogou/data/LUNA16',
                        help='Path to dataset')
    parser.add_argument('--b', default=16, type=int, help='Batch size')
    parser.add_argument('--output', default='None', type=str, help='Output path')
    parser.add_argument('--n', default='luna', choices=['luna', 'lidc', 'brats', 'lits'], type=str, help='Dataset to use')
    parser.add_argument('--workers', default=4, type=int, help='Num of workers')
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='GPU indices to use')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--k', default=10, type=int, help='Number of clusters for clustering pretask')
    parser.add_argument('--cpu', action='store_true', default=False, help='To run on CPU or not')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print(args)
    print()


def generate_gt(args, data_loader):

    if not args.gpu:
        cudnn.benchmark = True
        cudnn.deterministic = True

    # Generate colors for cluster masks
    palette = sns.color_palette(palette='bright', n_colors=args.k)
    colors = torch.Tensor([list(color) for color in palette]).cpu()  # cpu because we apply it on a detached tensor later

    # Get dataloaders
    train_loader = data_loader['train']
    
    # Get models
    featup = torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=False)
    featup = nn.DataParallel(featup)
    if not args.cpu:
        featup = featup.cuda()
    
    kmeans = KMeans(n_clusters=args.k, seed=args.seed)



    


    
        




    if not args.cpu:
        torch.cuda.empty_cache()
