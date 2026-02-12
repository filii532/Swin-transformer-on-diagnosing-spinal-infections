import os
import numpy as np
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
import re
from util import trainset_sample
from typing import Optional, Union
import torch
import pandas as pd
from organ_inf_list import *

class MyDataSet(Dataset):
    def __init__(self, path: list, ill_t: list, mode: str):
        self.path = path
        self.ill_t = ill_t
        self.mode = mode

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        # print(1)
        img = np.load(self.path[item])
        img, label = torch.tensor(img['image']) / 255., img['mask']
        img = torch.stack([img, img, img], axis=0)
        label = self.ill_t[item] if np.sum(label) > 20 else 0.
        label = torch.tensor(label).type(torch.LongTensor)

        if self.mode == 'train':
            return img, label
        elif self.mode == 'validation':
            (_, path) = os.path.split(self.path[item])
            return img, label, path.split("_")[0]
        elif self.mode == 'predict':
            (_, path) = os.path.split(self.path[item])
            (path, _) = os.path.splitext(path)
            return img, label, path
        


class MyDataModule(pl.LightningDataModule):
    def __init__(self,
                 path: Union[str, list],
                 batch_size: int = 32,
                 num_workers: int = 0,
                 num_classes: int = None,
                 **kwargs,
                 ):
        super().__init__()

        self.train_data = None
        self.val_data = None
        self.predict_data = None

        self.paths = [path] if isinstance(path, str) else path
        self.num_classes = len(path)+1 if num_classes==None else num_classes

        self.batch_size = batch_size
        self.num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, num_workers]) if num_workers else 0

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        train_patient, val_patient, predict_patient = train_, val_, []
        train_path, val_path, predict_path = [], [], []
        train_ill, val_ill, predict_ill = [], [], []

        if stage == 'fit':
            for idx, path in enumerate(self.paths):
                smp = [os.path.join(path, s) for s in os.listdir(path) if s[:10] in val_patient]
                val_path += smp
                val_ill = val_ill + [idx+1]*len(smp)

                p = [os.path.join(path, s) for s in os.listdir(path) if s[:10] in train_patient and s[-5] == '1']
                n = [os.path.join(path, s) for s in os.listdir(path) if s[:10] in train_patient and s[-5] == '0']
                smp = trainset_sample(p, n, mode='up')
                train_path = np.append(train_path, smp)
                train_ill = train_ill + [idx+1]*len(smp)
            
            self.train_patient, self.val_patient = train_patient, val_patient

            self.train_data = MyDataSet(path=train_path, 
                                        ill_t=train_ill,  
                                        mode='train')
            self.val_data = MyDataSet(path=val_path,
                                      ill_t=val_ill,  
                                      mode='validation')
            
        if stage == 'predict':
            for idx, path in enumerate(self.paths):
                smp = [os.path.join(path, s) for s in os.listdir(path)]# if s[:10] in all_
                predict_path += smp
                predict_ill = predict_ill + [idx+1]*len(smp)
            predict_path.sort()

            self.predict_data = MyDataSet(path=predict_path,
                                          ill_t=predict_ill,  
                                          mode='predict')

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=self.num_workers, )

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=self.num_workers, )

    def predict_dataloader(self):
        return DataLoader(self.predict_data,
                          batch_size=1,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=self.num_workers, )
    
    def val_patients(self):
        patients = ""
        for i in self.val_patient:
            patients = patients + f"{i}, "
        return patients

    def __len__(self):
        return len(self.train_path)
    
if __name__ == '__main__':
    pass