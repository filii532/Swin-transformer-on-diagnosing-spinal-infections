import numpy as np
import os
from utils import dice, check_dim, create_folder
import matplotlib.pyplot as plt
from tqdm import tqdm

data_path = r'/data/npz_data/bone/inf/trans/data'
pic_dir = r'/data/npz_data/bone/inf/trans/jpg/t1_t2'
create_folder(pic_dir, True)
modalities = {}

paths = [[s, f'{s[:10]}_2_{s[13:]}'] for s in os.listdir(data_path) if s[11] == '1' and os.path.exists(f'{data_path}/{s[:10]}_2_{s[13:]}')]

double = []
error = []
bar = tqdm(paths, ncols=100)
for path1, path2 in bar:
    bar.desc = path1

    img1 = np.load(os.path.join(data_path, path1))
    img2 = np.load(os.path.join(data_path, path2))
    img1, msk1 = img1["image"], img1["mask"]
    img2, msk2 = img2["image"], img2["mask"]


    if not check_dim(img1.shape, img2.shape):
        if img1.shape[0] == 2*img2.shape[0]:
            double.append(path1)
        elif 2*img1.shape[0] == img2.shape[0]:
            double.append(path2)
        else:
            error.append([path1, path2])
        continue

    if dice(msk1, msk2) < .9:
        for j in range((img1.shape[0]+1)//2):
            plt.figure(figsize=(20,20))
            fig, axs = plt.subplots(2, 2)
            ax1, ax2, ax3, ax4 = axs.flat

            img = img1[j]
            ax1.imshow(img, cmap='gray')
            if np.sum(msk1[j]):
                ax1.contour(msk1[j])
            ax1.axis('off')
            ax1.set_title("T1")
            
            img = img1[-j]
            ax3.imshow(img, cmap='gray')
            if np.sum(msk1[-j]):
                ax3.contour(msk1[-j])
            ax3.axis('off')
            ax3.set_title("T1 Reverse")

            img = img2[j]
            ax2.imshow(img, cmap='gray')
            if np.sum(msk2[j]):
                ax2.contour(msk2[j])
            ax2.axis('off')
            ax2.set_title("T2")

            img = img2[-j]
            ax4.imshow(img, cmap='gray')
            if np.sum(msk2[-j]):
                ax4.contour(msk2[-j])
            ax4.axis('off')
            ax4.set_title("T2 Reverse")

            plt.tight_layout()
            plt.savefig(os.path.join(pic_dir, f"{path1[:10]}_{path1[-5]}_{str(j).zfill(3)}.jpg"))
            plt.close('all')

# print("Doubled Image:")
# for i in double:
#     print(i)

# print("\nError Image:")
# for i in error:
#     print(i)