import os
import numpy as np
from utils import read_dicom, read_nrrd, create_folder, check_dim
import pandas as pd
from tqdm import tqdm
from process_list import pydicom_list, clahe_list, double_list, reverse_list

data_path = r'/data/raw_data/bone_inf/data_eff'
csv_path = r'/data/raw_data/bone_inf/list.csv'
save_path = r'/data/npz_data/bone/inf/trans/data'
create_folder(save_path, True)

csv = pd.read_csv(csv_path)
img_paths = list(csv["img"])
msk_paths = list(csv["mask"])
npz_paths = list(csv["path"])
error = []

bar = tqdm(zip(img_paths, msk_paths, npz_paths), total=len(img_paths), ncols=100)
for img_path, msk_path, npz_path in bar:
    bar.desc = npz_path

    # if npz_path != r'2300033380_2_0.npz':
    #     continue

    # if npz_path[:14] not in clahe_list:
    #     continue

    if_pydicom = 0
    if npz_path[:14] in pydicom_list:
        if_pydicom = 1
    # else:
    #     continue

    r = 1
    if npz_path[:12] in reverse_list:
        r = -1

    if os.path.exists(os.path.join(save_path, npz_path)):
        img = np.load(os.path.join(save_path, npz_path))
        img, msk = img["image"], img["mask"]
        msk += read_nrrd(os.path.join(data_path, msk_path))[::r]
        msk[msk > 0] = 1
    else:
        img = read_dicom(os.path.join(data_path, img_path), if_pydicom=if_pydicom)[::r]
        msk = read_nrrd(os.path.join(data_path, msk_path))[::r]
        if not check_dim(img.shape, msk.shape):
            img = read_dicom(os.path.join(data_path, img_path), if_del=True, if_pydicom=if_pydicom)
    if not check_dim(img.shape, msk.shape):
        error.append(npz_path)
        continue

    if npz_path[:14] in double_list:
        msk[msk.shape[0]//2:] += msk[:msk.shape[0]//2]
        msk[:msk.shape[0]//2] += msk[msk.shape[0]//2:]
        msk[msk > 0] = 1

    np.savez(os.path.join(save_path, npz_path), image=img, mask=msk)

for i in error:
    print(i)