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
import torchio.transforms

class LidcFineTune(Dataset):
    def __init__(self, config, img_list, train=False, crop_size=(64, 64, 64)):
        self.config = config
        self.train = train
        self.img_list = img_list
        self.crop_size = crop_size

        # Create setup file for pylidc
        txt = f"""
        [dicom]
        path = {config.data}
        warn = True
        """
        with open(os.path.join(os.path.expanduser('~'),'.pylidcrc'), 'w') as file:
            file.write(txt)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        pid = self.img_list[index]
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        ann = pl.query(pl.Annotation).filter(pl.Scan.patient_id == pid).first()
        x = scan.to_volume()  # Image
        y = ann.boolean_mask() # Segmentation mask

        # TODO: Scale down 0.5
        
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

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())