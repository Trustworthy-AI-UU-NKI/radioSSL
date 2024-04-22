import copy
import random
import time
import os
import pandas as pd
import numpy as np
import torch
import pylidc as pl
from PIL import Image
from scipy.special import comb
from torch.utils.data import Dataset
from torch.nn import functional as f
import torchio.transforms

class LidcFineTune(Dataset):
    def __init__(self, config, img_list, crop_size=(128, 128, 64), train=False): 
        # crop_size[1] = 64 because our data only has 94 slices (after preprocessing)
        self.config = config
        self.train = train
        self.img_list = img_list
        self.crop_size = crop_size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        pid = self.img_list[index]
        
        mask_path = os.path.join(self.config.data,pid,pid.replace('-','_') + '_raw.npy')
        seg_path = os.path.join(self.config.data,pid,pid.replace('-','_') + '_seg.npy')

        x = torch.from_numpy(np.load(mask_path))
        y = torch.from_numpy(np.load(seg_path))

        # Resize from 512 to 192
        x = x.T.unsqueeze(1) # Move slice dim to batch dim and add temporary channel dimension (H x W x D) -> (D x 1 x H x W)
        y = y.T.unsqueeze(1)
        x = f.interpolate(x.float(), size=(192,192))  # Scale only height and weight, not slice dim
        y = f.interpolate(y.float(), size=(192,192)).int()  
        x = x.permute(1,2,3,0)  # Put slice dim last (D x 1 x H x W -> 1 x H x W x D)
        y = y.permute(1,2,3,0)
        
        x, y = self.aug_sample(x, y)

        # min max
        x = self.normalize(x)

        return x, y
    
    def aug_sample(self, x, y):
        if self.train:
            # Random crop and augment
            x, y = self.random_crop(x, y)
            if random.random() < 0.5:
                x = torch.flip(x, dims=(1,))  # torch.flip not the source of the major slowdown
                y = torch.flip(y, dims=(1,))
            if random.random() < 0.5:
                x = torch.flip(x, dims=(2,))
                y = torch.flip(y, dims=(2,))
            if random.random() < 0.5:
                x = torch.flip(x, dims=(3,))
                y = torch.flip(y, dims=(3,))

        else:
            # Do not center crop for LIDC ()
            x, y = self.center_crop(x, y)
            pass
        
        return x, y

    def random_crop(self, x, y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def center_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())