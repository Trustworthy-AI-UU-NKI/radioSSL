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

    def __init__(self, config, img_train, train=False, transform=None, global_transforms=None, local_transforms=None):
        self.config = config
        self.imgs = img_train
        self.train = train
        self.transform = transform if transform != None else torchio.transforms.Compose([])
        self.global_transforms = global_transforms if global_transforms != None else torchio.transforms.Compose([])
        self.local_input_enable = (config.model != 'cluster')  # Do not include local_views in dataloader for cluster_pretask (TODO: might change later)
        self.local_transforms = local_transforms if local_transforms != None else torchio.transforms.Compose([])
        self.norm = torchio.transforms.ZNormalization()
        if config.model == 'cluster' :
            self.coords = pd.read_csv(os.path.join(config.data,'crop_coords.csv'), names=['path','crop1','crop2'], index_col='path')  # coordinates of each pair of crops
        else:
            self.coords = None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_path = self.imgs[index]
        relative_image_path = os.path.join(*os.path.normpath(image_path).split(os.sep)[-3:])
        pair = np.load(image_path)
        crop1 = pair[0]
        crop1 = np.expand_dims(crop1, axis=0)
        crop2 = pair[1]
        crop2 = np.expand_dims(crop2, axis=0)
        
        crop1_coords = []
        crop2_coords = []
        if self.config.model == 'cluster':
            crop1_coords = np.array(eval(self.coords.loc[relative_image_path]['crop1']))
            crop2_coords = np.array(eval(self.coords.loc[relative_image_path]['crop2']))

        input1 = self.transform(crop1)
        input2 = self.transform(crop2)
        gt1 = copy.deepcopy(input1)
        gt2 = copy.deepcopy(input2)
        input1 = self.global_transforms(input1)
        input2 = self.global_transforms(input2)

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
            torch.tensor(gt2, dtype=torch.float), crop1_coords, crop2_coords, local_inputs

class LitsFineTune(Dataset):
    def __init__(self, ct_dir, seg_dir, crop_size=(128, 128, 128), train=False, ratio=1.0):
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
        ct_array = f.interpolate(ct_array, scale_factor=(0.5,0.5))  # Scale only height and weight, not slice dim
        seg_array = f.interpolate(seg_array, scale_factor=(0.5,0.5))
        ct_array = ct_array.squeeze(1)
        seg_array = seg_array.squeeze(1)

        # random crop slices
        if self.train:
            start_slice = random.randint(0, ct_array.shape[0] - self.crop_size[2])
            end_slice = start_slice + self.crop_size[2] - 1
            ct_array = ct_array[start_slice:end_slice + 1, :, :]
            seg_array = seg_array[start_slice:end_slice + 1, :, :]
        else:
            start_slice = (ct_array.shape[0] - self.crop_size[2]) // 2
            end_slice = start_slice + self.crop_size[2] - 1
            ct_array = ct_array[start_slice:end_slice + 1, :, :]
            seg_array = seg_array[start_slice:end_slice + 1, :, :]
        # random crop height and width
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
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1]]

        return crop_volume, crop_seg