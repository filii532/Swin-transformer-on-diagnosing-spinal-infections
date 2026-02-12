# # !usr/bin/env python
# # -*- coding:utf-8 _*-
import os
import shutil
import numpy as np
from utils import create_folder, del_layer
from tqdm import tqdm
from process_list import del_dict

data_path = r'/data/npz_data/bone/inf/trans/data'
del_path = r'/data/npz_data/bone/inf/trans/del'
create_folder(del_path, True)


bar = tqdm(os.listdir(data_path), ncols=100)
for i in bar:
    bar.desc = i
    if i[:14] in del_dict.keys():
        data = np.load(os.path.join(data_path, i))
        image, mask = data['image'], data['mask']
        image, mask = del_layer(image, del_dict[i[:14]]), del_layer(mask, del_dict[i[:14]])
        shutil.copy(os.path.join(data_path, i), os.path.join(del_path, i))
        np.savez(os.path.join(data_path, i), image=image, mask=mask)

        
# for i in os.listdir(data_path):
#     if i[:14] in del_dict.keys() and i[11] in ['1', '2']:
#         data = np.load(os.path.join(data_path, i))
#         image, mask = data['image'], data['mask']
#         if not isinstance(del_dict[i[:14]][0], list):
#             if (del_dict[i[:14]][1] - del_dict[i[:14]][0] + 1) * 2 == image.shape[0]:
#                 print(i)

