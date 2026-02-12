import os
from tqdm import tqdm
from util import create_folder
import shutil
import numpy as np

label_path = r'F:\yqy\bone\data_yolo\labels'
pic_path = r'F:\yqy\bone\data_jpg\ct'
save_path = r'F:\yqy\bone\data_yolo\dataset'
create_folder(save_path)
create_folder(os.path.join(save_path, r'images\train'))
create_folder(os.path.join(save_path, r'images\test'))
create_folder(os.path.join(save_path, r'labels\train'))
create_folder(os.path.join(save_path, r'labels\test'))

bar = tqdm(os.listdir(label_path), ncols=100)
for path in bar:
    bar.desc = path
    with open(os.path.join(label_path, path), 'r') as f:
        l = f.readline()

    if l.split(" ")[0] == "0":
        continue
    l = " ".join(["0"]+l.split(" ")[1:])

    with open(os.path.join(label_path, path), 'w') as f:
        f.writelines(l)

patients = np.array(list(set([i[:10] for i in os.listdir(pic_path)])))
np.random.shuffle(patients)
paths = [i[:-4] for i in os.listdir(pic_path)]

for idx, path in tqdm(enumerate(paths), ncols=100, total=len(paths)):
    mod = "test" if path[:10] in patients[:int(len(patients)*.1)] else "train"
    
    shutil.copy(f'{pic_path}\\{path}.jpg', f'{save_path}\\images\\{mod}\\{path}.jpg')
    shutil.copy(f'{label_path}\\{path}.txt', f'{save_path}\\labels\\{mod}\\{path}.txt')

