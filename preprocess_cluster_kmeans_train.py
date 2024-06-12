import argparse
import os
import warnings
import sys
import time
import math
import numpy as np
import random
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import autocast

from faiss import Kmeans
from sklearn.decomposition import PCA

from data import DataGenerator
from tools import set_seed


warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Training benchmark')
    parser.add_argument('--data', metavar='DIR', default='/home/igogou/data/LUNA16',
                        help='Path to dataset')
    parser.add_argument('--ratio', default=1, type=float, help='Ratio of data used for pretraining/finetuning.')
    parser.add_argument('--model', default='cluster', choices=['cluster'], help='Choose the model')
    parser.add_argument('--upsampler', default='featup', choices=['featup','interp'], help='Choose the model')
    parser.add_argument('--b', default=16, type=int, help='Batch size')
    parser.add_argument('--b_kmeans', default=16, type=int, help='Batch size of kmeans')
    parser.add_argument('--n', default='luna', choices=['luna', 'lidc', 'brats', 'lits'], type=str, help='Dataset to use')
    parser.add_argument('--workers', default=4, type=int, help='Num of workers')
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='GPU indices to use')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--k', default=10, type=int, help='Number of clusters for clustering pretask')
    parser.add_argument('--cpu', action='store_true', default=False, help='To run on CPU or not')
    args = parser.parse_args()
    print(args)
    print()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Set seed
    set_seed(args.seed)
    print(f'Seed is {args.seed}\n')

    # Force arguments
    args.model = 'cluster'
    args.ratio = 1

    # Get dataloader
    generator = DataGenerator(args)
    loader_name = 'cluster_' + args.n + '_pretask'
    train_loader = getattr(generator, loader_name)(load_gt=False)['train']
    train_loader.shuffle = False
    
    # Get models
    featup = nn.DataParallel(torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=False))
    if not args.cpu:
        featup = featup.cuda()
    else:
        featup = featup.cpu()
    kmeans = Kmeans(d=384, k=args.k, seed=args.seed, verbose=False, gpu=True if not args.cpu else False)

    featup.eval()


    # Train K-Means ----------------------------------------------------------------------------------
    print('Train K-Means...')

    with tqdm(total=len(train_loader)) as tqdm_progress:
        for idx, (input1, input2, _, _, _, _, _, index) in enumerate(train_loader):

            print(f'Iteration {idx}/{len(train_loader)}', flush=True)

            input1 = input1.float()
            input2 = input2.float()

            if not args.cpu:
                input1 = input1.cuda()
                input2 = input2.cuda()

            # Convert 3D input to 2D
            B, C, H, W, D = input1.shape
            x1 = input1.permute(0,4,1,2,3).reshape(B*D,C,H,W)  # B x C x H x W x D -> B*D x C x H x W
            x2 = input2.permute(0,4,1,2,3).reshape(B*D,C,H,W)

            with torch.no_grad():
                
                print('Featup', flush=True)
                # Get upsampled features from teacher DINO ViT16 encoder and flatten spatial dimensions to get feature vectors for each pixel
                # B*D x 1 x H x W -(RGB)->  B*D x 3 x H x W -(Featup)-> B*D x C' x H x W -(Vectorize)-> B*D*H*W x C' 
                # feat_vec1 = torch.zeros((B*D,384,H,W))  # C' = 384
                # feat_vec2 = torch.zeros((B*D,384,H,W))
                # MB = 8 # Mini-batch size (to work on my local machine)
                if args.upsampler == 'featup':
                    # for b_idx in tqdm(range(0,B*D,MB), leave=False):  
                    #     feat_vec1[b_idx:b_idx+MB] = featup.module(x1[b_idx:b_idx+MB].repeat(1,3,1,1))
                    #     feat_vec2[b_idx:b_idx+MB] = featup.module(x2[b_idx:b_idx+MB].repeat(1,3,1,1))
                    feat_vec1 = featup.module(x1.repeat(1,3,1,1))  # Put it through DINO and FeatUP
                    feat_vec2 = featup.module(x2.repeat(1,3,1,1))
                elif args.upsampler == 'interp':
                    # for b_idx in tqdm(range(0,B*D,MB), leave=False): 
                    #     feat_vec1[b_idx:b_idx+MB] = f.interpolate(featup.module.model(x1[b_idx:b_idx+MB].repeat(1,3,1,1)), size=(H,W), mode='bilinear')
                    #     feat_vec2[b_idx:b_idx+MB] = f.interpolate(featup.module.model(x2[b_idx:b_idx+MB].repeat(1,3,1,1)),  size=(H,W), mode='bilinear')
                    feat_vec1 = f.interpolate(featup.module.model(x1.repeat(1,3,1,1)), size=(H,W), mode='bilinear')  # Put it only through DINO and upsample with interpolation
                    feat_vec2 = f.interpolate(featup.module.model(x2.repeat(1,3,1,1)), size=(H,W), mode='bilinear')
                feat_vec1 = feat_vec1.permute(0,2,3,1).flatten(0,2)
                feat_vec2 = feat_vec2.permute(0,2,3,1).flatten(0,2)

                # Train K-Means
                print('K-Means', flush=True)
                kmeans.train(torch.cat([feat_vec1,feat_vec2]).cpu().numpy())

            tqdm_progress.update(1)

    kmeans_path = os.path.join(args.data,f'kmeans_centroids_k{args.k}_{args.upsampler}.npy')
    np.save(kmeans_path,kmeans.centroids)
    print(f'Centroids saved at: {kmeans_path}')
    