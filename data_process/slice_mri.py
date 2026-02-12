import numpy as np
from utils import create_folder
from tqdm import tqdm
import os
from scipy.ndimage import zoom

data_dir = r'/data/npz_data/bone/inf/trans/data'
t1_dir = r'/data/npz_data/bone/inf/slice/t1_sag'
t2_dir = r'/data/npz_data/bone/inf/slice/t2_sag'
t3_dir = r'/data/npz_data/bone/inf/slice/t2_tra'
create_folder(t1_dir)
create_folder(t2_dir)
create_folder(t3_dir)

img_size = [224., 224.]

bar = tqdm(os.listdir(data_dir), ncols=100)
for data_path in bar:
    bar.desc = data_path
    if data_path[11] == "0":
        continue
    data = np.load(os.path.join(data_dir, data_path))
    image, mask = data['image'], data['mask']
    image = zoom(image, [1, img_size[0]/image.shape[1], img_size[1]/image.shape[2]], order=3)
    mask = zoom(mask, [1, img_size[0]/mask.shape[1], img_size[1]/mask.shape[2]], order=1)
    
    mod, data_path = data_path[11], f"{data_path[:10]}{data_path[12:-4]}"
    for i in range(image.shape[0]):
        img, msk = image[i], mask[i]
        np.savez(os.path.join(eval(f't{mod}_dir'), f'{data_path}_{str(i).zfill(3)}'), image=img, mask=msk)