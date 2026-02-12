import pandas as pd
from utils import del_layer
from tqdm import tqdm
import SimpleITK as sitk
from process_list import del_dict, pydicom_list
import pydicom
import os
import numpy as np
import warnings
warnings.filterwarnings("error")

csv_path = r'/data/raw_data/bone_inf/list.csv'
data_path = r'/data/raw_data/bone_inf/data_eff'
npz_path = r'/data/npz_data/bone/trans/data'

csv = pd.read_csv(csv_path)
img_paths = list(csv["img"])
npz_paths = list(csv["path"])

ori_error = {}
loc_error = {}
key_error = []
bar = tqdm(zip(img_paths, npz_paths), total=len(img_paths), ncols=100)
for img, path in bar:
    bar.desc = path
    paths = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(os.path.join(data_path, img))
    lim = np.load(os.path.join(npz_path, f"{path}.npz"))["mask"]
    lim = np.where(lim > 0)
    x_min, x_max = np.min(lim[0]), np.max(lim[0])+1
    
    try:
        if path in pydicom_list:
            slices = [pydicom.dcmread(s) for s in paths]
            if path in del_dict.keys():
                slices = del_layer(slices, del_dict[path])
            slices = slices[x_min:x_max]
            orien = [s[0x00200037].value for s in slices]
            loc = [float(s[0x00201041].value) for s in slices]
        else:
            slices = [sitk.ReadImage(s) for s in paths]
            if path in del_dict.keys():
                slices = del_layer(slices, del_dict[path])
            slices = slices[x_min:x_max]
            orien = [s.GetMetaData("0020|0037") for s in slices]
            loc = [float(s.GetMetaData("0020|1041")) for s in slices]
    except KeyError as e:
        key_error.append(path)
        continue

    orien = np.array([i.split("\\") for i in orien] if isinstance(orien[0], str) else orien, dtype=np.float32)
    orien = np.unique(orien, axis=0)
    if len(orien) > 1:
        uni = np.ones([len(orien)])
        for i in range(6):
            data = np.squeeze(orien[:, i])
            uni = np.concatenate(([True], ~np.isclose(data[1:], data[:-1]))) * uni
        orien = orien[uni.astype(bool)]

    if len(loc) > 1:
        loc.sort()
        loc = np.unique(np.diff(loc))
        loc = loc[np.concatenate(([True], ~np.isclose(loc[1:], loc[:-1])))]
        loc = loc[loc != 0.]

    if len(orien) > 1:
        ori_error[path] = orien

    if len(loc) > 1:
        loc_error[path] = loc

if len(ori_error.keys()) > 0:
    print('Varying Orientation:')
    for patient in ori_error.keys():
        print(f"{patient}: {ori_error[patient]}")

if len(loc_error.keys()) > 0:
    print('Varying Slice Thickness:')
    for patient in loc_error.keys():
        print(f"{patient}: {loc_error[patient]}")

if len(key_error) > 0:
    print('Key Missing:')
    for patient in key_error:
        print(f"{patient}")

print("\n\n")
if len(ori_error.keys()) + len(loc_error.keys()) > 0:
    patients = list(set(list(ori_error.keys()) + list(loc_error.keys())))
    for patient in patients:
        print(patient)

