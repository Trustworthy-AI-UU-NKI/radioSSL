import warnings
from skimage.transform import resize

warnings.filterwarnings('ignore')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import sys
import random
import csv

import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from optparse import OptionParser
from glob import glob
from multiprocessing import Pool

sys.setrecursionlimit(40000)

parser = OptionParser()
parser.add_option("--n", dest="n", help='dataset to use', default="luna", choices=['luna', 'lits', 'brats'], type="choice")
parser.add_option("--fold", dest="fold", help="fold of subset", default=None, type="int")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type="int")
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type="int")
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type="int")
parser.add_option("--crop_rows", dest="crop_rows", help="crop rows", default=64, type="int")
parser.add_option("--crop_cols", dest="crop_cols", help="crop cols", default=64, type="int")
parser.add_option("--lung_max", dest="lung_max", help="lung max", default=0.15, type="float")
parser.add_option("--data", dest="data", help="the directory of the dataset", default='/data/LUNA16',
                  type="string")
parser.add_option("--save", dest="save", help="the directory of processed 3D cubes",
                  default=None, type="string")
parser.add_option("--scale", dest="scale", help="scale of the generator", default=16, type="int")
parser.add_option('--z_align', action='store_true', dest='z_align', default=False, help='z dim align when cropping')
parser.add_option('--seed', default=1, type="int")

(options, args) = parser.parse_args()
fold = options.fold
seed = options.seed
random.seed(seed)
print(f'Seed: {seed}')

assert options.data is not None
assert options.save is not None
# assert options.fold >= 0 and options.fold <= 9

if not os.path.exists(options.save):
    os.makedirs(options.save)


class setup_config():
    hu_max = 1000.0
    hu_min = -1000.0
    HU_thred = (-150.0 - hu_min) / (hu_max - hu_min)

    def __init__(self,
                 input_rows=None,
                 input_cols=None,
                 input_deps=None,
                 crop_rows=None,
                 crop_cols=None,
                 len_border=None,
                 len_border_z=None,
                 scale=None,
                 DATA_DIR=None,
                 SAVE_DIR=None,
                 train_fold=[0, 1, 2, 3, 4],
                 valid_fold=[5, 6],
                 test_fold=[7, 8, 9],
                 len_depth=None,
                 lung_min=0.7,
                 lung_max=1.0,
                 z_align = False
                 ):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_deps = input_deps
        self.crop_rows = crop_rows
        self.crop_cols = crop_cols
        self.len_border = len_border
        self.len_border_z = len_border_z
        self.scale = scale
        self.DATA_DIR = DATA_DIR
        self.train_fold = train_fold
        self.valid_fold = valid_fold
        self.test_fold = test_fold
        self.len_depth = len_depth
        self.lung_min = lung_min
        self.lung_max = lung_max
        self.SAVE_DIR = SAVE_DIR
        self.z_align = z_align

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


config = setup_config(input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      crop_rows=options.crop_rows,
                      crop_cols=options.crop_cols,
                      scale=options.scale,
                      len_border=70,
                      len_border_z=15,
                      len_depth=3,
                      lung_min=0.7,
                      lung_max=options.lung_max,
                      DATA_DIR=options.data,
                      SAVE_DIR=options.save,
                      z_align=options.z_align
                      )
config.display()

col_size = [(96, 96, 64), (96, 96, 96), (112, 112, 64), (64, 64, 32)]
col_size_z_align = [(96, 96, 32), (96, 96, 32), (112, 112, 32), (64, 64, 32)]  # If z_align then the slice dim must always be the same size 
input_rows, input_cols, input_depth = (64, 64, 32)
local_col_size = [(32, 32, 16), (16, 16, 16), (32, 32, 32), (8, 8, 8)]
local_input_rows, local_input_cols, local_input_depth = (16, 16, 16)


def load_sitk_with_resample(img_path):
    outsize = [0, 0, 0]
    outspacing = [1, 1, 1]

    # 读取文件的size和spacing信息
    vol = sitk.ReadImage(img_path)
    tmp = sitk.GetArrayFromImage(vol)
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()

    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.5)
    outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.5)
    outsize[2] = int(inputsize[2] * inputspacing[2] / outspacing[2] + 0.5)

    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol


def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, xmax1, ymin1, ymax1, zmin1, zmax1 = box1
    xmin2, xmax2, ymin2, ymax2, zmin2, zmax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1) * (zmax1 - zmin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2) * (zmax2 - zmin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    zmin = max(zmin1, zmin2)
    zmax = min(zmax1, zmax2)
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    d = max(0, zmax - zmin)
    area = w * h * d  # C∩G的面积
    iou = area / (s1 + s2 - area)
    return iou


def crop_pair(img_array, z_align = False):

    while True:
        size_x, size_y, size_z = img_array.shape
        # print(img_array.shape)
        img_array1 = img_array.copy()
        img_array2 = img_array.copy()
        if size_z - 64 - config.len_depth - 1 - config.len_border_z < config.len_border_z:
            pad = size_z - 64 - config.len_depth - 1 - config.len_border_z - config.len_border_z
            padding = [0, 0, -pad + 1]
            img_array1 = np.pad(img_array1, padding, mode='constant', constant_values=0)

        if size_z - 64 - config.len_depth - 1 - config.len_border_z < config.len_border_z:
            pad = size_z - 64 - config.len_depth - 1 - config.len_border_z - config.len_border_z
            padding = [0, 0, -pad + 1]
            img_array2 = np.pad(img_array2, padding, mode='constant', constant_values=0)
            size_z += -pad + 1
        if z_align:
            crop_size = col_size_z_align
        else:
            crop_size = col_size
        while True:
            size_index1 = np.random.randint(0, len(crop_size))
            crop_rows1, crop_cols1, crop_deps1 = crop_size[size_index1]
            size_index2 = np.random.randint(0, len(crop_size))
            crop_rows2, crop_cols2, crop_deps2 = crop_size[size_index2]
            if size_x - crop_rows1 - 1 - config.len_border <= config.len_border:
                crop_rows1 -= 32
                crop_cols1 -= 32
            if size_x - crop_rows2 - 1 - config.len_border <= config.len_border:
                crop_rows2 -= 32
                crop_cols2 -= 32
            start_x1 = random.randint(0 + config.len_border, size_x - crop_rows1 - 1 - config.len_border)
            start_y1 = random.randint(0 + config.len_border, size_y - crop_cols1 - 1 - config.len_border)
            start_z1 = random.randint(0 + config.len_border_z,
                                      size_z - crop_deps1 - config.len_depth - 1 - config.len_border_z)
            start_x2 = random.randint(0 + config.len_border, size_x - crop_rows2 - 1 - config.len_border)
            start_y2 = random.randint(0 + config.len_border, size_y - crop_cols2 - 1 - config.len_border)
            if z_align:  # If z_align, the crops must be on the same slices (because there is only 2D RoI align)
                start_z2 = start_z1
            else:
                start_z2 = random.randint(0 + config.len_border_z,
                                        size_z - crop_deps2 - config.len_depth - 1 - config.len_border_z)          
            crop_coords1 = (start_x1, start_x1 + crop_rows1, start_y1, start_y1 + crop_cols1, start_z1, start_z1 + crop_deps1)
            crop_coords2 = (start_x2, start_x2 + crop_rows2, start_y2, start_y2 + crop_cols2, start_z2, start_z2 + crop_deps2)
            iou = cal_iou(crop_coords1, crop_coords2)
            if iou > 0.3:
                break

        crop_window1 = img_array1[start_x1: start_x1 + crop_rows1,
                       start_y1: start_y1 + crop_cols1,
                       start_z1: start_z1 + crop_deps1 + config.len_depth,
                       ]

        crop_window2 = img_array2[start_x2: start_x2 + crop_rows2,
                       start_y2: start_y2 + crop_cols2,
                       start_z2: start_z2 + crop_deps2 + config.len_depth,
                       ]

        if crop_rows1 != input_rows or crop_cols1 != input_cols or crop_deps1 != input_depth:
            crop_window1 = resize(crop_window1,
                                  (input_rows, input_cols, input_depth + config.len_depth),
                                  preserve_range=True,
                                  )
        if crop_rows2 != input_rows or crop_cols2 != input_cols or crop_deps2 != input_depth:
            crop_window2 = resize(crop_window2,
                                  (input_rows, input_cols, input_depth + config.len_depth),
                                  preserve_range=True,
                                  )
        t_img1 = np.zeros((input_rows, input_cols, input_depth), dtype=float)
        d_img1 = np.zeros((input_rows, input_cols, input_depth), dtype=float)
        t_img2 = np.zeros((input_rows, input_cols, input_depth), dtype=float)
        d_img2 = np.zeros((input_rows, input_cols, input_depth), dtype=float)
        for d in range(input_depth):
            for i in range(input_rows):
                for j in range(input_cols):
                    for k in range(config.len_depth):
                        if crop_window1[i, j, d + k] >= config.HU_thred:
                            t_img1[i, j, d] = crop_window1[i, j, d + k]
                            d_img1[i, j, d] = k
                            break
                        if k == config.len_depth - 1:
                            d_img1[i, j, d] = k
        for d in range(input_depth):
            for i in range(input_rows):
                for j in range(input_cols):
                    for k in range(config.len_depth):
                        if crop_window2[i, j, d + k] >= config.HU_thred:
                            t_img2[i, j, d] = crop_window2[i, j, d + k]
                            d_img2[i, j, d] = k
                            break
                        if k == config.len_depth - 1:
                            d_img2[i, j, d] = k

        d_img1 = d_img1.astype('float32')
        d_img1 /= (config.len_depth - 1)
        d_img1 = 1.0 - d_img1
        d_img2 = d_img2.astype('float32')
        d_img2 /= (config.len_depth - 1)
        d_img2 = 1.0 - d_img2

        if np.sum(d_img1) > config.lung_max * crop_cols1 * crop_deps1 * crop_rows1:
            continue
        # print(np.sum(d_img1))
        if np.sum(d_img2) > config.lung_max * crop_cols1 * crop_deps1 * crop_rows1:
            continue
        # we start to crop the local windows
        x_min = min(crop_coords1[0], crop_coords2[0])
        x_max = max(crop_coords1[1], crop_coords2[1])
        y_min = min(crop_coords1[2], crop_coords2[2])
        y_max = max(crop_coords1[3], crop_coords2[3])
        z_min = min(crop_coords1[4], crop_coords2[4])
        z_max = max(crop_coords1[5], crop_coords2[5])
        local_windows = []
        for i in range(6):

            local_x = np.random.randint(max(x_min - 3, 0), min(x_max + 3, size_x))
            local_y = np.random.randint(max(y_min - 3, 0), min(y_max + 3, size_y))
            local_z = np.random.randint(max(z_min - 3, 0), min(z_max + 3, size_z))
            local_size_index = np.random.randint(0, len(local_col_size))
            local_crop_rows, local_crop_cols, local_crop_deps = local_col_size[local_size_index]
            local_window = img_array1[local_x: local_x + local_crop_rows,
                           local_y: local_y + local_crop_cols,
                           local_z: local_z + local_crop_deps
                           ]
                #if local_crop_rows != local_input_rows or local_crop_cols != local_input_cols or local_crop_deps != local_input_depth:
            local_window = resize(local_window,
                                  (local_input_rows, local_input_cols, local_input_depth),
                                  preserve_range=True,
                                  )
            local_windows.append(local_window)

        return crop_window1[:, :, :input_depth], crop_window2[:, :, :input_depth], np.stack(local_windows, axis=0), crop_coords1, crop_coords2


def infinite_generator_from_one_volume(img_array, save_dir, root_dir, name):
    split_root_dir = os.path.normpath(root_dir).split(os.sep)
    split_save_dir = os.path.normpath(save_dir).split(os.sep)
    if split_root_dir == split_save_dir:
        relative_save_dir = ''
    else:
        relative_save_dir = os.path.join(*[folder for folder in split_save_dir if folder not in split_root_dir])

    csv_lines = []
    
    img_array[img_array < config.hu_min] = config.hu_min
    img_array[img_array > config.hu_max] = config.hu_max
    img_array = 1.0 * (img_array - config.hu_min) / (config.hu_max - config.hu_min)
    for num_pair in range(config.scale):
        crop_window1, crop_window2, local_windows, crop_coords1, crop_coords2 = crop_pair(img_array, z_align=config.z_align)
        crop_window = np.stack((crop_window1, crop_window2), axis=0)
        global_name = name + '_global_' + str(num_pair) + '.npy'
        local_name = name + '_local_' + str(num_pair)  + '.npy'
        global_path = os.path.join(save_dir, global_name)
        local_path = os.path.join(save_dir, local_name)
        np.save(global_path, crop_window)
        np.save(local_path, local_windows)
        relative_save_path = os.path.join(relative_save_dir, global_name)
        csv_lines.append([relative_save_path,crop_coords1,crop_coords2])

    return csv_lines


def brats_preprocess():
    save_path = config.SAVE_DIR

    csv_file = open(os.path.join(save_path,'crop_coords.csv'), 'w', newline='\n')
    csv_writer = csv.writer(csv_file)
    
    for subset in ['HGG', 'LGG']:
        print(">> Subset {}".format(subset))
        brats_subset_path = os.path.join(config.DATA_DIR, subset)
        folder_list = os.listdir(os.path.join(brats_subset_path))
        for folder in tqdm(folder_list, desc='Folders parsed'):
            file_list = glob(os.path.join(brats_subset_path, folder, "*.nii.gz"))  # Only selects .nii.gz files (DOES NOT exclude the segmentations)
            save_dir = os.path.join(save_path, subset, folder)
            os.makedirs(save_dir, exist_ok=True)
            for img_file in tqdm(file_list, desc='Images in folder parsed', leave=False):                
                img_name = os.path.split(img_file)[-1]
                if 'seg' not in img_name:  # Skip segmentation masks
                    img_array = load_sitk_with_resample(img_file)
                    img_array = sitk.GetArrayFromImage(img_array)
                    img_array = img_array.transpose(2, 1, 0)
                    img_csv_rows = infinite_generator_from_one_volume(img_array=img_array, save_dir=save_dir, root_dir=save_path, name=img_name[:-4])  # remove file type .mhd (4 chars)
                    csv_writer.writerows(img_csv_rows)
                    csv_file.flush()

    csv_file.close()


def luna_preprocess_thread(fold):
    save_path = config.SAVE_DIR

    for index_subset in fold:
        csv_file = open(os.path.join(save_path,f'crop_coords_{index_subset}.csv'), 'w', newline='\n')
        csv_writer = csv.writer(csv_file)
        print(">> Fold {}".format(index_subset))
        luna_subset_path = os.path.join(config.DATA_DIR, "subset" + str(index_subset))
        file_list = glob(os.path.join(luna_subset_path, "*.mhd"))  # Only selects mhd files (excludes the segmentations)
        subset = 'subset' + str(index_subset)
        save_dir = os.path.join(save_path, subset)
        
        os.makedirs(save_dir, exist_ok=True)
        for img_file in tqdm(file_list, desc=f'Images in fold {index_subset} parsed'):
            img_name = os.path.split(img_file)[-1]
            img_array = load_sitk_with_resample(img_file)
            img_array = sitk.GetArrayFromImage(img_array)
            img_array = img_array.transpose(2, 1, 0)
            img_csv_rows = infinite_generator_from_one_volume(img_array=img_array, save_dir=save_dir, root_dir=save_path, name=img_name[:-4])  # remove file type .mhd (4 chars)
            csv_writer.writerows(img_csv_rows)
            csv_file.flush()

        csv_file.close()


def luna_preprocess():

    # Multi-thread preprocess
    with Pool(10) as p:
        p.map(luna_preprocess_thread, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    
    # Combine csv files
    save_path = config.SAVE_DIR
    final_file = open(os.path.join(save_path,f'crop_coords.csv'), 'w', newline='\n')
    writer = csv.writer(final_file)
    files = [open(os.path.join(save_path,f'crop_coords_{i}.csv'), 'r', newline='\n') for i in range(10)]
    for file in files:
        reader = csv.reader(file)
        for row in reader:
            writer.writerow(row)
    final_file.close()


# Main execution    
if options.n == 'luna':
    # assert options.lung_max == 0.30
    luna_preprocess()
elif options.n == 'brats':
    # assert options.lung_max == 1
    brats_preprocess()

    

