import cv2
import numpy as np
import os
from util import create_folder
from organ_tumor_list import all_
from tqdm import tqdm
os.system('clear')

src_dir = r'/data/npz_data/bone/tumor/3d'
src_path = [f'{src_dir}/{i}' for i in os.listdir(src_dir) if i[11]=='0']

save_path_ori = r'/data/doctor/ori'
save_path_aux = r'/data/doctor/aux'
create_folder(save_path_ori)
create_folder(save_path_aux)

for path in tqdm(src_path, ncols=100):
    data = np.load(path)
    image, mask = data['image'], data['mask']

    path = os.path.basename(path).split('.')[0].split('_')
    if path[0] not in all_:
        continue
    path = f'{path[0]}_{path[2]}'

    for i in range(2):
        idxs = np.arange(image.shape[0])
        np.random.shuffle(idxs)
        idxs = idxs[:20]

        img, msk = image[idxs], [1 if np.sum(msk) > 0 else 0 for msk in mask[idxs]]
        np.savez(fr'{save_path_ori}/{path}_{i}.npz', image=img, mask=msk)

        img = np.array([cv2.imread(fr'/data/bone_tumor_pic/{path}_{str(idx).zfill(3)}.jpg', 0) for idx in idxs])
        np.savez(fr'{save_path_aux}/{path}_{i}.npz', image=img, mask=msk)

