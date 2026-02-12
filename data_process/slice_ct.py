import numpy as np
from utils import create_folder
import os
from tqdm import tqdm
from scipy.ndimage import zoom
from matplotlib import pyplot as plt

ori_path = r'/data/npz_data/bone/inf/trans/data'
label_path = r'/data/npz_data/bone/inf/yolo_label/ct_tra'
save_path = r'/data/npz_data/bone/inf/slice/ct_tra'
# pic_path = r'F:\yqy\bone\dataset\slice\ct_img'
create_folder(save_path)
# create_folder(pic_path)
img_size = [224., 224.]

paths = [i[:-4] for i in os.listdir(ori_path) if i[11] == '0']
bar = tqdm(paths, ncols=100, total=len(paths))
for path in bar:
    bar.desc = path

    img = np.load(os.path.join(ori_path, f'{path}.npz'))
    img, msk = img["image"], img["mask"]
    l, x, y = img.shape

    for i in range(l):
        with open(os.path.join(label_path, f'{path}_{str(i).zfill(3)}.txt'), "r") as f:
            loc = f.readline()
        loc = [float(j) for j in loc.split(" ")[1:]]

        x1, y1, x2, y2 = int((loc[1]-.5*loc[3])*x), int((loc[0]-.5*loc[2])*y), int((loc[1]+.5*loc[3])*x), int((loc[0]+.5*loc[2])*y)

        image = zoom(img[i][x1:x2, y1:y2], [img_size[0]/(x2-x1), img_size[1]/(y2-y1)], order=3)
        mask = zoom(msk[i][x1:x2, y1:y2], [img_size[0]/(x2-x1), img_size[1]/(y2-y1)], order=1)

        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        # plt.figure(figsize=(10, 10))
        # plt.axis('off')
        # plt.imshow(image, cmap='gray')
        # if np.sum(mask) > 0:
        #     plt.contour(mask, linewidths=2)
        # plt.savefig(os.path.join(pic_path, f'{path}_{str(i).zfill(3)}.jpg'), bbox_inches='tight', pad_inches=0)
        # plt.close()

        np.savez(os.path.join(save_path, f'{path[:10]}_{path[13]}_{str(i).zfill(3)}_{1 if np.sum(mask) else 0}.npz'), image=image, mask=mask)
    
