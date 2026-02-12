import os
from utils import create_folder
import shutil

pic_path = r'F:\yqy\bone\data_jpg\ct'
cpr_path = r'F:\yqy\bone\data_process\yolo\runs\detect\13\crops'
save_path = r'F:\yqy\bone\data_yolo'
create_folder(os.path.join(save_path, 'rewrite'))

rewrite_list = list(set(os.listdir(pic_path)) - set(os.listdir(cpr_path)))
for i in rewrite_list:
    shutil.copy(os.path.join(pic_path, i), os.path.join(save_path, 'rewrite', i))