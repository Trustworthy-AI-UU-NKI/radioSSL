import copy
import random
import time
import os
import numpy as np
import torch
from PIL import Image
from scipy.special import comb
from torch.utils.data import Dataset
import torchio.transforms
import SimpleITK as sitk
from torch.nn import functional as f


class LitsFineTune(Dataset):
    def __init__(self, ct_dir, seg_dir, training, ratio=1.0):
        self.crop_size = [128, 128]  # height and width dimensions (after cropping)
        self.size = 64  # slice dimension size 
        self.training = training
        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(
            map(lambda x: x.replace('volume', 'segmentation').replace('.nii', '.nii.gz'), self.ct_list))

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))
        self.ct_list = self.ct_list[: int(len(self.ct_list) * ratio)]
        self.seg_list = self.seg_list[: int(len(self.seg_list) * ratio)]

    def __getitem__(self, index):
        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        ct_array = torch.FloatTensor(ct_array)
        seg_array = torch.FloatTensor(seg_array)

        # resize
        ct_array = ct_array.unsqueeze(1)  # Add temporary channel dimension
        seg_array = seg_array.unsqueeze(1)
        ct_array = f.interpolate(ct_array, scale_factor=(0.5,0.5))  # Scale only height and weight, not slice dim
        seg_array = f.interpolate(seg_array, scale_factor=(0.5,0.5))
        ct_array = ct_array.squeeze(1)
        seg_array = seg_array.squeeze(1)

        # random crop slice
        if self.training:
            start_slice = random.randint(0, ct_array.shape[0] - self.size)
            end_slice = start_slice + self.size - 1
            ct_array = ct_array[start_slice:end_slice + 1, :, :]
            seg_array = seg_array[start_slice:end_slice + 1, :, :]
        else:
            start_slice = (ct_array.shape[0] - self.size) // 2
            end_slice = start_slice + self.size - 1
            ct_array = ct_array[start_slice:end_slice + 1, :, :]
            seg_array = seg_array[start_slice:end_slice + 1, :, :]
        ct_array, seg_array = self.random_crop(ct_array, seg_array)

        ct_array = ct_array.unsqueeze(0)
        seg_array = seg_array.unsqueeze(0)

        ct_array = ct_array.permute(0,2,1,3)  # 1, S, H, W -> 1 x H x S x W
        seg_array = seg_array.permute(0,2,1,3) 

        # min max
        ct_array = (ct_array - ct_array.min()) / (ct_array.max() - ct_array.min())

        return ct_array, seg_array

    def __len__(self):
        return len(self.ct_list)

    def random_crop(self, x, y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        depth, height, width = x.shape[-3:]
        sx = random.randint(0, height - crop_size[-2] - 1)
        sy = random.randint(0, width - crop_size[-1] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1]]

        return crop_volume, crop_seg