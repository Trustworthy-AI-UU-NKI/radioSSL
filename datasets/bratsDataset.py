import os
import random
import copy
import pandas as pd

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio.transforms



class Pcrlv2BraTSPretask(Dataset):

    def __init__(self, config, img_train, train=False, transform=None, global_transforms=None, local_transforms=None):
        self.config = config
        self.imgs = img_train
        self.train = train
        self.transform = transform
        self.global_transforms = global_transforms
        self.local_input_enable = (config.phase != 'cluster_pretask')  # Do not include local_views in dataloader for cluster_pretask (TODO: might change later)
        self.local_transforms = local_transforms
        self.norm = torchio.transforms.ZNormalization()
        self.coords = pd.read_csv(os.path.join(config.data,'crop_coords.csv'), names=['path','crop1','crop2'], index_col='path')  # coordinates of each pair of crops

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
            # local_inputs = []
            for i in range(locals.shape[0]):
                img = locals[i]
                img = np.expand_dims(img, axis=0)
                img = self.transform(img)
                img = self.local_transforms(img)
                local_inputs.append(img)

        return torch.tensor(input1, dtype=torch.float), torch.tensor(input2, dtype=torch.float), \
            torch.tensor(gt1, dtype=torch.float), \
            torch.tensor(gt2, dtype=torch.float), crop1_coords, crop2_coords, local_inputs


class BratsFineTune(Dataset):

    def __init__(self, patients_dir, crop_size=(112, 128, 112), modes=("t1", "t2", "flair", "t1ce"), train=True):
        self.patients_dir = patients_dir
        self.modes = modes
        self.train = train
        self.crop_size = crop_size

    def __len__(self):
        return len(self.patients_dir)

    def __getitem__(self, index):
        patient_dir = self.patients_dir[index]
        volumes = []
        modes = list(self.modes) + ['seg']
        for mode in modes:
            patient_id = os.path.split(patient_dir)[-1]
            volume_path = os.path.join(patient_dir, patient_id + "_" + mode + '.nii.gz')
            volume = nib.load(volume_path).get_data()
            if not mode == "seg":
                volume = self.normlize(volume)  # [0, 1.0]
            volumes.append(volume)  # [h, w, d]
        seg_volume = volumes[-1]
        volumes = volumes[:-1]
        volume, seg_volume = self.aug_sample(volumes, seg_volume)
        wt_volume = seg_volume > 0  # 坏死和无增强的肿瘤区域：1、增强区域（活跃部分）：4、周边水肿区域：2
        tc_volume = np.logical_or(seg_volume == 4, seg_volume == 1)
        et_volume = (seg_volume == 4)
        seg_volume = [wt_volume, tc_volume, et_volume]
        seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        return (torch.tensor(volume.copy(), dtype=torch.float),
                torch.tensor(seg_volume.copy(), dtype=torch.float))

    def aug_sample(self, volumes, mask):
        """
            Args:
                volumes: list of array, [h, w, d]
                mask: array [h, w, d], segmentation volume
            Ret: x, y: [channel, h, w, d]

        """
        x = np.stack(volumes, axis=0)  # [N, H, W, D][w, h, d] [d, h, w][2, 1, 0]
        y = np.expand_dims(mask, axis=0)  # [channel, h, w, d]

        if self.train:
            # crop volume
            x, y = self.random_crop(x, y)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if random.random() < 0.5:
                x = np.flip(x, axis=3)
                y = np.flip(y, axis=3)
        else:
            x, y = self.center_crop(x, y)

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

    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())
