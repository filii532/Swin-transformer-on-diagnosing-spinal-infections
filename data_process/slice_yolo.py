import numpy as np
from utils import create_folder
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

ori_path = r'/data/npz_data/bone/inf/trans/data'
save_path = r'/home/ubuntu/data_process/bone_inf/yolo/images'
create_folder(save_path)

paths = [i[:-4] for i in os.listdir(ori_path) if i[11] == '0']
bar = tqdm(paths, ncols=100, total=len(paths))
for path in bar:
    bar.desc = path

    img = np.load(os.path.join(ori_path, f'{path}.npz'))
    img, msk = img["image"], img["mask"]
    l, x, y = img.shape

    for i in range(l):
        if f'{path}_{str(i).zfill(3)}.txt' in os.listdir(r'/data/npz_data/bone/inf/yolo_label/ct_tra'):
            continue

        image = img[i]
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = np.array(image, dtype=np.uint8)
        
        plt.imsave(f'{save_path}/{path}_{str(i).zfill(3)}.jpg', image, cmap='gray')
    
