import copy
import random
import time
import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from scipy.special import comb
from torch.utils.data import Dataset
import torchio.transforms


class LunaPretask(Dataset):

    def __init__(self, config, img_train, train=False, transform=None, global_transforms=None, local_transforms=None, load_gt=True):
        self.config = config
        self.imgs = img_train
        self.train = train
        self.transform = transform if transform != None else torchio.transforms.Compose([])
        self.global_transforms = global_transforms if global_transforms != None else torchio.transforms.Compose([])
        self.local_input_enable = (config.model != 'cluster')  # Do not include local_views in dataloader for cluster_pretask (TODO: might change later)
        self.local_transforms = local_transforms if local_transforms != None else torchio.transforms.Compose([])
        self.norm = torchio.transforms.ZNormalization()
        self.global_index = [0, 1, 2, 3, 4, 5, 6, 7]
        self.local_index = [i for i in range(48)]
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
        relative_image_path = os.path.join(*os.path.normpath(image_path).split(os.sep)[-2:])
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
                gt_path = name + f"_gt_k{self.config.k}" + ext
                gt_pair = np.load(image_path)
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
            for i in range(locals.shape[0]):
                img = locals[i]
                img = np.expand_dims(img, axis=0)
                img = self.transform(img)
                img = self.local_transforms(img)
                local_inputs.append(img)

        return torch.tensor(input1, dtype=torch.float), torch.tensor(input2, dtype=torch.float), \
            torch.tensor(gt1, dtype=torch.float), \
            torch.tensor(gt2, dtype=torch.float), crop1_coords, crop2_coords, local_inputs, index

    def bernstein_poly(self, i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """

        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    def bezier_curve(self, points, nTimes=1000):
        """
           Given a set of control points, return the
           bezier curve defined by the control points.

           Control points should be a list of lists, or list of tuples
           such as [ [1,1],
                     [2,3],
                     [4,5], ..[Xn, Yn] ]
            nTimes is the number of time steps, defaults to 1000

            See http://processingjs.nihongoresources.com/bezierinfo/
        """

        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([self.bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals

    def data_augmentation(self, x, y, prob=0.5):
        # augmentation by flipping
        cnt = 3
        while random.random() < prob and cnt > 0:
            degree = random.choice([0, 1, 2])
            x = np.flip(x, axis=degree)
            y = np.flip(y, axis=degree)
            cnt = cnt - 1

        return x, y

    def nonlinear_transformation(self, x, prob=0.5):
        if random.random() >= prob:
            return x
        points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
        xpoints = [p[0] for p in points]
        ypoints = [p[1] for p in points]
        xvals, yvals = self.bezier_curve(points, nTimes=100000)
        if random.random() < 0.5:
            # Half change to get flip
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        nonlinear_x = np.interp(x, xvals, yvals)
        return nonlinear_x

    def local_pixel_shuffling(self, x, prob=0.5, num_block=10000):
        if random.random() >= prob:
            return x
        image_temp = copy.deepcopy(x)
        orig_image = copy.deepcopy(x)
        _, img_rows, img_cols, img_deps = x.shape
        for _ in range(num_block):
            block_noise_size_x = random.randint(1, img_rows // 10)
            block_noise_size_y = random.randint(1, img_cols // 10)
            block_noise_size_z = random.randint(1, img_deps // 10)
            noise_x = random.randint(0, img_rows - block_noise_size_x)
            noise_y = random.randint(0, img_cols - block_noise_size_y)
            noise_z = random.randint(0, img_deps - block_noise_size_z)
            window = orig_image[0, noise_x:noise_x + block_noise_size_x,
                     noise_y:noise_y + block_noise_size_y,
                     noise_z:noise_z + block_noise_size_z,
                     ]
            window = window.flatten()
            np.random.shuffle(window)
            window = window.reshape((block_noise_size_x,
                                     block_noise_size_y,
                                     block_noise_size_z))
            image_temp[0, noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = window
        local_shuffling_x = image_temp

        return local_shuffling_x

    def image_in_painting(self, x, cnt=5):
        _, img_rows, img_cols, img_deps = x.shape
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)
            block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)
            block_noise_size_z = random.randint(img_deps // 6, img_deps // 3)
            noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = np.random.rand(block_noise_size_x,
                                                                   block_noise_size_y,
                                                                   block_noise_size_z, ) * 1.0
            cnt -= 1
        return x

    def image_out_painting(self, x, cnt=4):
        _, img_rows, img_cols, img_deps = x.shape
        image_temp = copy.deepcopy(x)
        x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
        x[:,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y,
                                                noise_z:noise_z + block_noise_size_z]
        while cnt > 0 and random.random() < 0.95:
            block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
            block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
            block_noise_size_z = img_deps - random.randint(3 * img_deps // 7, 4 * img_deps // 7)
            noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                    noise_y:noise_y + block_noise_size_y,
                                                    noise_z:noise_z + block_noise_size_z]
            cnt -= 1
        return x


def augmentation(x, config):
    # flip
    cnt = 4
    while random.random() < config.flip_rate and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        cnt = cnt - 1
    # rotate
    if random.random() < config.rotate_rate:
        x = np.rot90(x, k=random.randint(0, 3))
    # gauss noise
    if random.random() < config.gauss_rate:
        x = x + np.random.normal(0, 1, x.shape)
    # transpose ?

    return x


class LunaFineTune(Dataset):
    def __init__(self, config, true_list, false_list, train, input_shape=(64, 64, 64), test=False):
        self.config = config
        self.input_shape = input_shape
        self.true_list = true_list
        self.true_list = self.true_list + self.true_list
        self.false_list = false_list
        self.train = train
        self.test = test
        # self.candidate = self.true_list + self.false_list
        # self.multi = len(self.false_list) // len(self.true_list)

    def __len__(self):
        if self.train:
            return len(self.true_list) * 3
        else:
            return len(self.true_list) + len(self.false_list)
        # return len(self.candidate)

    def __getitem__(self, index):
        # if self.train:
        #     if index % 3 == 0:
        #         path = self.true_list[index // 3]
        #         img_array = np.load(path)
        #         label = 1
        #         # if self.train:
        #         #     img_array = augmentation(img_array, self.config)
        #     else:
        #         rand = random.randint(0, len(self.false_list) - 1)
        #         path = self.false_list[rand]
        #         img_array = np.load(path)
        #         label = 0
        #     img_array = np.expand_dims(img_array, axis=0)
        #     return torch.from_numpy(img_array.copy()).float(),label
        # else:
        if index >= len(self.true_list):
            path = self.false_list[index - len(self.true_list)]
            img_array = np.load(path)
            label = 0
        else:
            path = self.true_list[index]
            img_array = np.load(path)
            label = 1
        # if self.train:
        #     img_array = augmentation(img_array, self.config)
        img_array = np.expand_dims(img_array, axis=0)
        return torch.from_numpy(img_array.copy()).float(), torch.from_numpy(seg_array.copy()).float(), label