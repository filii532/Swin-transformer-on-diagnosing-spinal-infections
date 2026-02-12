import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import create_folder

data_dir = r'/data/npz_data/bone/inf/slice'
contour_dir = r'/data/npz_data/bone/inf/trans/jpg/contour'
create_folder(contour_dir, True)

for data_folder in os.listdir(data_dir):
    create_folder(fr'{contour_dir}/{data_folder}')
    bar = tqdm(os.listdir(fr'{data_dir}/{data_folder}'), ncols=100)
    for data_path in bar:
        bar.desc = data_path

        data = np.load(fr'{data_dir}/{data_folder}/{data_path}')
        image = data['image']
        mask = data['mask']

        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(image, cmap='gray')
        if np.sum(mask) > 0:
            plt.contour(mask, linewidths=2)
        plt.savefig(fr'{contour_dir}/{data_folder}/{data_path[:-4]}.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()

    
