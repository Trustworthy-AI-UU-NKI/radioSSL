import os
import random
import numpy as np
from tqdm import tqdm
import pylidc as pl

random.seed(1)
np.random.seed(1)

root_folder = '/projects/0/prjs0905/data/LIDC'
save_folder = '/projects/0/prjs0905/data/LIDC_proc'

# Create setup file for pylidc
txt = f"""
[dicom]
path = {root_folder}
warn = True
"""
with open(os.path.join(os.path.expanduser('~'),'.pylidcrc'), 'w') as file:
    file.write(txt)

paths = []

for folder_path, _, file_names in os.walk(root_folder):
    rel_folder_path = os.path.relpath(folder_path,root_folder)
    depth = rel_folder_path.count(os.path.sep)
    if depth == 0 and 'LIDC-IDRI' in rel_folder_path:
        paths.append(rel_folder_path.split('/')[-1])

# Save images and segmentation masks as .npy files
for pid in tqdm(paths):
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    ann = pl.query(pl.Annotation).filter(pl.Scan.patient_id == pid).first()
    
    # Image
    try:
        x = scan.to_volume()
    except Exception as e:
        raise RuntimeError(f"Corrupted file in {pid}. Redownload!") from e

    # Segmentation mask
    y = np.zeros(x.shape)

    mask = ann.boolean_mask()
    bbox = ann.bbox()
    y[bbox[0].start:bbox[0].stop,bbox[1].start:bbox[1].stop,bbox[2].start:bbox[2].stop] = mask
    print(bbox[2].start,bbox[2].stop)

    # Put the tumor center slice in the center slice of the image and keep only surrounding slices 
    # (We do this to make sure all samples have the same slice number because in the raw data it varies.)
    # (We also do this so that the crops we take in the dataloader have a higher chance to contain the tumor)
    tumor_center_slice = (bbox[2].start + bbox[2].stop) // 2 
    slice_span = min(x.shape[-1] - tumor_center_slice, tumor_center_slice)
    y = y[:,:,tumor_center_slice - slice_span:tumor_center_slice + slice_span]
    x = x[:,:,tumor_center_slice - slice_span:tumor_center_slice + slice_span]

    sample_path = os.path.join(save_folder,pid)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    mask_path = os.path.join(sample_path,pid.replace('-','_') + '_raw.npy')
    seg_path = os.path.join(sample_path,pid.replace('-','_') + '_seg.npy')
    
    np.save(mask_path, x)
    np.save(seg_path, y)
