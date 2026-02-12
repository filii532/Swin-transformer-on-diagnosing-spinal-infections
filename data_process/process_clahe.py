import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import create_folder
from enhance.clahe import CLAHE
from scipy.ndimage import zoom
img_size = [512., 512.]

data_dir = r'/data/npz_data/bone/inf/trans/data'
pic_dir = r'/data/npz_data/bone/inf/trans/jpg/clahe'
create_folder(pic_dir, True)

bar = tqdm(os.listdir(data_dir), ncols=100)
for path in bar:
    bar.desc = path
    data = np.load(os.path.join(data_dir, path))
    image, mask = data['image'], data['mask']
    image = zoom(image, [1, img_size[0]/image.shape[1], img_size[1]/image.shape[2]], order=3)
    mask = zoom(mask, [1, img_size[0]/mask.shape[1], img_size[1]/mask.shape[2]], order=1)

    new = []
    for i in range(image.shape[0]):
        img = image[i]
        equalized_image = CLAHE(img, 32)
        new.append(equalized_image)
        plt.figure(figsize=(10,20))

        plt.subplot(1,2,1)
        plt.imshow(img, cmap='gray')
        if np.sum(mask[i]):
            plt.contour(mask[i])
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(equalized_image, cmap='gray')
        if np.sum(mask[i]):
            plt.contour(mask[i])
        plt.axis('off')

        plt.savefig(os.path.join(pic_dir, path[:-4] + '_' + str(i).zfill(3) + '.jpg'))
        plt.close()
    np.savez(os.path.join(data_dir, path), image=new, mask=mask)