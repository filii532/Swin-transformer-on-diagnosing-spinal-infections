import os
import pandas as pd
import shutil
from process_list import del_list, pair_list
from utils import create_folder
from collections import Counter
from tqdm import tqdm

all_data = pd.DataFrame()

src_dir = r'/data/raw_data/bone_inf/data'
dst_dir = r'/data/raw_data/bone_inf/data_eff'
save_path = r'/data/raw_data/bone_inf/list.csv'
create_folder(dst_dir)

replacement_dict = {'删除': 4, 'ct': 0, '矢t1': 1, '矢t2': 2, '横t2': 3}

for folder in tqdm([fr'{src_dir}/{i}' for i in os.listdir(src_dir) if os.path.isdir(fr'{src_dir}/{i}')], ncols=100):
    del_data = pd.DataFrame()
    if os.path.exists(fr'{folder}/不符合数据.xlsx'):
        del_data = pd.concat((del_data, pd.read_excel(fr'{folder}/不符合数据.xlsx', index_col=0))).reset_index(drop=True)
    if os.path.exists(fr'{folder}/重复数据.xlsx'):
        del_data = pd.concat((del_data, pd.read_excel(fr'{folder}/重复数据.xlsx', index_col=0))).reset_index(drop=True)
    
    csv = pd.read_csv(fr'{folder}/list.csv', encoding='GB2312', index_col=0)
    csv['label'].replace(replacement_dict, inplace=True)
    csv = csv[csv['label'].isin([0, 1, 2, 3])]
    csv = csv[~csv['mask'].isin(del_list)]
    csv = csv[~csv.isin(del_data.to_dict(orient='list')).all(axis=1)]

    csv.insert(0, 'patient', list(map(lambda x: str(x).zfill(10), csv['Patient'])))
    csv = csv.drop(['SeriesInstanceUID', 'path', 'url', 'Patient'], axis=1)
    csv = csv.sort_values(by=['patient', 'label'], ascending=[True, True]).reset_index(drop=True)

    for img, mask in zip(csv['img'], csv['mask']):
        if os.path.exists(fr'{folder}/{mask}') and not os.path.exists(fr'{dst_dir}/{mask}') :
            if not os.path.exists(fr'{dst_dir}/{img}'):
                shutil.copytree(fr'{folder}/{img}', fr'{dst_dir}/{img}')
            shutil.copy2(fr'{folder}/{mask}', fr'{dst_dir}/{mask}')
        else:
            csv = csv[csv['mask'] != mask]

    all_data = pd.concat((all_data, csv))

all_data = all_data.sort_values(by=['patient', 'label'], ascending=[True, True]).reset_index(drop=True)

n_rois = [0]
time = 0
for line in range(1, all_data.shape[0]):
    if all_data['img'][line] in pair_list:
        n_rois.append(pair_list[all_data['img'][line]])
    elif all_data['patient'][line] != all_data['patient'][line-1]:
        time = 0
        n_rois.append(0)
    elif all_data['label'][line] != all_data['label'][line-1]:
        time = 0
        n_rois.append(0)
    elif all_data['img'][line] == all_data['img'][line-1]:
        n_rois.append(time)
    else:
        time += 1
        n_rois.append(time)
all_data['n_rois'] = n_rois

path = []
for line in range(all_data.shape[0]):
    path.append(f"{all_data['patient'][line]}_{all_data['label'][line]}_{all_data['n_rois'][line]}")
all_data["path"] = path
# print([item for item, count in Counter(all_data['mask']).items() if count > 1])

all_data.to_csv(save_path, index=False, encoding='utf-8')