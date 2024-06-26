import copy
import random
import time
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import comb
from torch.utils.data import Dataset
import torchio.transforms
import SimpleITK as sitk
from torch.nn import functional as f


class LitsPretask(Dataset):

    def __init__(self, config, img_train, train=False, transform=None, global_transforms=None, local_transforms=None, load_gt=True):
        self.config = config
        self.imgs = img_train
        self.train = train
        self.transform = transform if transform != None else torchio.transforms.Compose([])
        self.global_transforms = global_transforms if global_transforms != None else torchio.transforms.Compose([])
        self.local_input_enable = (config.model != 'cluster')  # Do not include local_views in dataloader for cluster_pretask (TODO: might change later)
        self.local_transforms = local_transforms if local_transforms != None else torchio.transforms.Compose([])
        self.norm = torchio.transforms.ZNormalization()
        self.load_gt = load_gt
        if 'cluster' in config.model:
            self.coords = pd.read_csv(os.path.join(config.data,'crop_coords.csv'), names=['path','crop1','crop2'], index_col='path')  # coordinates of each pair of crops
        else:
            self.coords = None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        # Load data
        image_path = self.imgs[index]
        relative_image_path = os.path.join(*os.path.normpath(image_path).split(os.sep)[-3:])
        pair = np.load(image_path)
        crop1 = pair[0]
        crop1 = np.expand_dims(crop1, axis=0)
        crop2 = pair[1]
        crop2 = np.expand_dims(crop2, axis=0)
        
        # Crop coordinates
        crop1_coords = []
        crop2_coords = []
        if 'cluster' in self.config.model:
            crop1_coords = np.array(eval(self.coords.loc[relative_image_path]['crop1']))
            crop2_coords = np.array(eval(self.coords.loc[relative_image_path]['crop2']))

        input1 = self.transform(crop1)
        input2 = self.transform(crop2)

        # Ground truth
        if 'cluster' in self.config.model:
            if self.load_gt:
                name, ext = os.path.splitext(image_path)
                gt_path = name + f"_gt_k{self.config.k}_{self.config.upsampler}" + ext  # TODO: Revert it back to this one, the other is just for debugging old files
                # gt_path = name + f"_gt_k{self.config.k}" + ext
                gt_pair = np.load(gt_path)
                gt1 = gt_pair[0]
                gt2 = gt_pair[1]
            else:
                gt1 = []
                gt2 = []
        elif self.config.model == 'pcrlv2':
            gt1 = copy.deepcopy(input1)
            gt2 = copy.deepcopy(input2)

        # Global input
        input1 = self.global_transforms(input1)
        input2 = self.global_transforms(input2)

        # Local input
        local_inputs = []
        if self.local_input_enable:
            locals = np.load(image_path.replace('global', 'local'))
            for i  in range(locals.shape[0]):
                img = locals[i]
                img = np.expand_dims(img, axis=0)
                img = self.transform(img)
                img = self.local_transforms(img)
                local_inputs.append(img)

        return torch.tensor(input1, dtype=torch.float), torch.tensor(input2, dtype=torch.float), \
            torch.tensor(gt1, dtype=torch.float), \
            torch.tensor(gt2, dtype=torch.float), crop1_coords, crop2_coords, local_inputs, index

class LitsFineTune(Dataset):
    def __init__(self, ct_dir, seg_dir, crop_size=(128, 128, 64), train=False, ratio=1.0):
        # cropped slices are 64 because our data has minimum 75 slices
        self.crop_size = crop_size
        self.train = train
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
        ct_array = ct_array.unsqueeze(1)  # Add temporary channel dimension (D x H x W -> (D x 1 x H x W)
        seg_array = seg_array.unsqueeze(1)
        ct_array = f.interpolate(ct_array, scale_factor=(0.5,0.5))  # Scale only height and weight, not slice dim  # TODO: Check that this doesn't flip height and width because finetune and pretrain datasets end up with diff orientation
        seg_array = f.interpolate(seg_array, scale_factor=(0.5,0.5))
        ct_array = ct_array.squeeze(1)
        seg_array = seg_array.squeeze(1)

        # Keep only lung class (class 1)
        seg_array[seg_array==2] = 1

        # random crop only for training, otherwise center crop
        depth, height, width = ct_array.shape[-3:]
        if self.train:
            x1 = random.randint(0, height - self.crop_size[0] - 1)
            y1 = random.randint(0, width - self.crop_size[1] - 1)
            z1 = random.randint(0, depth - self.crop_size[2] - 1)
        else:
            x1 = (height - self.crop_size[0] - 1) // 2
            y1 = (width - self.crop_size[1] - 1) // 2
            z1 = (depth - self.crop_size[2] - 1) // 2
        x2 = x1 + self.crop_size[0]
        y2 = y1 + self.crop_size[1]
        z2 = z1 + self.crop_size[2]
        ct_array = ct_array[z1:z2, x1:x2, y1:y2]
        seg_array = seg_array[z1:z2, x1:x2, y1:y2]

        ct_array = ct_array.unsqueeze(0)
        seg_array = seg_array.unsqueeze(0)

        ct_array = ct_array.permute(0,2,3,1)  # 1, D, H, W -> 1 x H x W x D
        seg_array = seg_array.permute(0,2,3,1) 

        # HU truncate
        ct_array[ct_array < -1000] = -1000
        ct_array[ct_array > 1000] = 1000

        # min max
        ct_array = (ct_array - ct_array.min()) / (ct_array.max() - ct_array.min())

        return ct_array, seg_array

    def __len__(self):
        return len(self.ct_list)