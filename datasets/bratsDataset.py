import os
import random
import copy
import pandas as pd

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio.transforms


class BratsPretask(Dataset):

    def __init__(self, config, img_train, train=False, transform=None, global_transforms=None, local_transforms=None, load_gt=True):
        self.config = config
        self.imgs = img_train
        self.train = train
        self.transform = transform if transform != None else torchio.transforms.Compose([])
        self.global_transforms = global_transforms if global_transforms != None else torchio.transforms.Compose([])
        self.local_input_enable = ('cluster' not in config.model)  # Do not include local_views in dataloader for cluster pretask (TODO: might change later)
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
        # relative_image_path = relative_image_path.replace('.ni','')  # TODO: This is just for debugging purposes because some old preprocessed datasets had a filename bug
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
        if self.config.model == 'cluster':  # Only cluster, not cluster_patch
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
        else:
            gt1 = []
            gt2 = []

        # Global input
        input1 = self.global_transforms(input1)
        input2 = self.global_transforms(input2)

        # Local input
        local_inputs = []
        if self.local_input_enable:
            locals = np.load(image_path.replace('global', 'local'))
            for i in range(locals.shape[0]):
                img = locals[i]
                img = np.expand_dims(img, axis=0)
                img = self.transform(img)
                img = self.local_transforms(img)
                local_inputs.append(img)

        return torch.tensor(input1, dtype=torch.float), torch.tensor(input2, dtype=torch.float), \
            torch.tensor(gt1, dtype=torch.float), \
            torch.tensor(gt2, dtype=torch.float), crop1_coords, crop2_coords, local_inputs, index


class BratsFineTune(Dataset):

    def __init__(self, patients_dir, crop_size=(128, 128, 128), modes=("t1", "t2", "flair", "t1ce"), train=True):
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
            volume = nib.load(volume_path).get_fdata()
            if not mode == "seg":
                volume = self.normalize(volume)  # [0, 1.0]
            volumes.append(volume)  # [h, w, d]
        seg_volume = volumes[-1]
        volumes = volumes[:-1]
        volume, seg_volume = self.aug_sample(volumes, seg_volume) # 1 x H x W x D
        wt_volume = seg_volume > 0 
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
            # if random.random() < 0.5:
            #     x = np.flip(x, axis=1)
            #     y = np.flip(y, axis=1)
            # if random.random() < 0.5:
            #     x = np.flip(x, axis=2)
            #     y = np.flip(y, axis=2)
            # if random.random() < 0.5:
            #     x = np.flip(x, axis=3)
            #     y = np.flip(y, axis=3)
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

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())
