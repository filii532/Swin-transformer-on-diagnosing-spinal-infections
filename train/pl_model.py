from typing import Any, Callable
import lightning.pytorch as pl
import torch
from math import cos, sqrt, pi
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
import numpy as np
import os
from sklearn import metrics
import model.model as m
import sys
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import pandas as pd
from util import create_folder
from tqdm import tqdm
import time
import cv2

from organ_inf_list import *

import matplotlib.pyplot as plt
import matplotlib
zhfont1 = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/msyh.ttc") 

class CT_Classification(pl.LightningModule):
    def __init__(
            self,
            model: str,
            num_classes: int = 2,
            lr_max: float = 1e-4,
            lr_min: float = 1e-7,
            transform=None,
            ts = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        model = model.lower().replace("-", "_")
        self.model = model
        dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        net = eval("m."+model)(num_classes=num_classes)

        if model+".pth" in os.listdir(os.path.join(dir, r"pretrain")):
            try:
                weights_dict = torch.load(os.path.join(
                    dir, fr"./pretrain/{model}.pth"))["model"]
            except:
                weights_dict = torch.load(os.path.join(
                    dir, fr"./pretrain/{model}.pth"))
            load_weights_dict = {k: v for k, v in weights_dict.items(
            ) if k in net.state_dict() if net.state_dict()[k].numel() == v.numel()}
            if load_weights_dict.keys() == []:
                raise KeyError("Pretrain Error")
            net.load_state_dict(load_weights_dict, strict=False)

        self.net = net

        self.transform = transform

        self.T = 15
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.ts = ts
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW

        self.train_gt = torch.tensor([])
        self.train_pr = torch.tensor([])
        self.val_gt = torch.tensor([])
        self.val_pr = torch.tensor([])
        self.val_pt = []

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = self.optimizer(params, lr=self.lr_max, weight_decay=5e-4)
        def lambda1(epoch): return ((cos((epoch % self.T)/self.T * pi)+1) /
                                    2 * (self.lr_max-self.lr_min)+self.lr_min)/self.lr_max
        scheduler = {
            'scheduler': LambdaLR(optimizer, lambda1),
            'name': 'learming_rate'
        }
        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_epoch_start(self) -> None:
        self.train_gt = torch.tensor([])
        self.train_pr = torch.tensor([])
        self.val_gt = torch.tensor([])
        self.val_pr = torch.tensor([])
        self.val_pt = []
        return super().on_train_epoch_start()
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        x = self.transform['train'](x)
        y_pre = self.forward(x)

        self.train_pr = torch.cat((self.train_pr, y_pre.detach().cpu()), dim=0)
        self.train_gt = torch.cat(
            (self.train_gt, torch.tensor(y.cpu(), dtype=torch.uint8)), dim=0)

        loss = self.loss(y_pre, y)
        self.log("train/loss", loss)

        return loss

    def on_train_epoch_end(self) -> None:
        loss = self.loss(self.train_pr, self.train_gt.long())
        self.log_dict({"train/loss_epoch": loss, "step": self.current_epoch})

    def validation_step(self, batch, batch_idx):
        x, y, paths = batch
        x = self.transform['val'](x)
        y_pre = self.forward(x)  # [:, 1].unsqueeze(1)

        self.val_pr = torch.cat((self.val_pr, y_pre.softmax(dim=1).cpu()[:, 1]), dim=0)
        self.val_gt = torch.cat((self.val_gt, torch.tensor(y.cpu(), dtype=torch.uint8)), dim=0)
        self.val_pt = np.append(self.val_pt, paths)

        loss = self.loss(y_pre, y)
        self.log("val/loss", loss)

    def on_validation_epoch_end(self) -> None:
        train_pr = nn.Softmax(dim=1)(self.train_pr)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(
            self.train_gt, train_pr, pos_label=1)
        threshold = thresholds[np.argmin(abs(tpr-(1-fpr)))]

        pred = self.val_pr > threshold
        tp = torch.sum(self.val_gt+pred == 2)
        tn = torch.sum(self.val_gt+pred == 0)
        fn = (torch.sum(self.val_gt) - torch.sum(self.val_gt+pred == 2))
        fp = (torch.sum(pred) - torch.sum(self.val_gt+pred == 2))
        fpr, tpr, thresholds = metrics.roc_curve(
            self.val_gt, self.val_pr, pos_label=1)

        self.log_dict({
            "threshold": threshold,
            "matrix/sensitivity": tp/(tp+fn)*100,
            "matrix/specificity": tn/(tn+fp)*100,
            "matrix/auc": metrics.auc(fpr, tpr)*100,
            "matrix/acc": (tp+tn)/(tp+tn+fp+fn)*100,
            "save/check": metrics.auc(fpr, tpr)+tp/(tp+fn)+tn/(tn+fp),
            "step": self.current_epoch
        })

        sen = []
        spe = []
        auc = []
        acc = []
        for patient in np.unique(self.val_pt):
            pr = self.val_pr[self.val_pt == patient] > threshold
            gt = self.val_gt[self.val_pt == patient]

            tp = torch.sum(gt+pr == 2)
            tn = torch.sum(gt+pr == 0)
            fn = (torch.sum(gt) - torch.sum(gt+pr == 2))
            fp = (torch.sum(pr) - torch.sum(gt+pr == 2))
            fpr, tpr, thresholds = metrics.roc_curve(gt, pr, pos_label=1)

            sen.append(tp/(tp+fn))
            spe.append(tn/(tn+fp))
            auc.append(metrics.auc(fpr, tpr))
            acc.append((tp+tn)/(tp+tn+fp+fn))

        self.log_dict({
            "matrix_per_patient/sensitivity": np.average(sen)*100,
            "matrix_per_patient/specificity": np.average(spe)*100,
            "matrix_per_patient/auc": np.average(auc)*100,
            "matrix_per_patient/acc": np.average(acc)*100,
            "step": self.current_epoch
        })

    def check_gradecam(self, dataloader):
        cam = GradCAM(model=self, target_layers=self.net.grad_cam_layer(), reshape_transform=self.net.reshape_transform)
        targets = [ClassifierOutputTarget(1)]
        create_folder("./grad_cam/")

        for image, gt, path in tqdm(dataloader, ncols=100):
            gt, path = gt.numpy()[0], path[0]
            if not os.path.exists(fr"./grad_cam/{path.split('_')[0]}"):
                create_folder(fr"./grad_cam/{path.split('_')[0]}/correct")
                create_folder(fr"./grad_cam/{path.split('_')[0]}/error")
                create_folder(fr"./grad_cam/{path.split('_')[0]}/origin")
            cam.batch_size = len(image)
            x = self.transform['predict'](image)
            x = x.to(self.device)
            pr = nn.Sigmoid()(self.forward(x)).detach().cpu().numpy()[0]
            pr = int(pr[1] > self.ts)

            image = image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
            grayscale_cam = cam(input_tensor=x, targets=targets, eigen_smooth=True)[0]

            plt.figure(figsize=(10, 10))
            plt.axis('off')
            plt.imshow(image)
            plt.savefig(fr'./grad_cam/{path.split("_")[0]}/origin/{path}.jpg', bbox_inches='tight', pad_inches=0)
            plt.close()
            
            plt.figure(figsize=(10, 10))
            plt.axis('off')
            visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
            plt.imshow(visualization)
            # plt.title(f'pred: {pr} label: {gt}')
            plt.savefig(fr'./grad_cam/{path.split("_")[0]}/{"correct" if pr==gt else "error"}/{path}.jpg', bbox_inches='tight', pad_inches=0)
            plt.close()

    def on_predict_epoch_start(self) -> None:
        self.pre_log = []
        self.pre_gt = torch.tensor([])
        self.pre_pr = torch.tensor([])
        self.pre_pt = []
        self.pre_tm = []
        self.record = []

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        start_time = time.time()
        x, y, paths = batch
        y_pre = self.forward(x)  # [:, 1].unsqueeze(1)
        self.record.append([f'{paths[0]}', round(float(y_pre.softmax(dim=1)[0, 1]), 3)])

        self.pre_pr = torch.cat((self.pre_pr, y_pre.softmax(dim=1).cpu()[:, 1]), dim=0)
        self.pre_gt = torch.cat((self.pre_gt, torch.tensor(y.cpu(), dtype=torch.uint8)), dim=0)
        self.pre_pt = np.append(self.pre_pt, paths)
        self.pre_tm = np.append(self.pre_tm, [(time.time()-start_time)/len(x)]*len(x), axis=0)
        

    def on_predict_epoch_end(self) -> None:

        # 机器识别的正确与不正确的数据发给我吗，以及机器识别每张图的时间
        patients = [    
            "0000584912",
            "0000795091",
            "0000967966",
            "0001791475",
            "0001851215",
            "0001935852",
            "0002403134",
            "0002458821",
            "0003750097",
            "0004106853",
            "0004286886",
            "0004441040",
            "0005331858",
            "0005517886",
            "0005521499",
            "0005560644",
            "0005620906",
            "0005686382",
            "0005741444",
            "0006017747",
            "2000121787",
            "2100013142",
            "2100859723",
            "2101033990",
            "2101092451",
            "2200049813",
            "2200138382",
            "2200275147",
            "2200296783",
            "2200432016",
            "2200543806",
            "2200700855",
            "2200708588",
            "2200754045",
            "2200766912",
            "2201077505",
            "2201144157",
            "2201214772",
            "2201239552",
            "2201487830",
            "2300040753",
            "2300139413",
            "2300165849",
            "2300248060",
            "2300309368",
            "2300382819",
            "2300385553",
            "2300464289",]
        
        df = {
            'patient': [],
            'index': [],
            'time': [],
            'output': [],
            'acc': [],
            '原始图片的判断结果': [],
            '机器识别的结果': []
            }
        
        for pr, gt, pt, tm in zip(self.pre_pr, self.pre_gt, self.pre_pt, self.pre_tm):
            
            df['patient'].append(pt[:12])
            df['index'].append(pt[13:16])
            df['acc'].append('正确' if int(pr>self.ts) == gt else '错误')
            df['output'].append(f'{pr:.4f}')
            df['原始图片的判断结果'].append('正确' if gt else '错误')
            df['机器识别的结果'].append('正确' if int(pr>self.ts) else '错误')
            df['time'].append(f'{tm:.4f}')
        df = pd.DataFrame(df)
        df.to_excel('output.xlsx', index=False)
        # 辅助判断
        # pr = {i: 1 if k > self.ts else 0 for i, k in zip(self.pre_pt, self.pre_pr)}
        # create_folder("/data/bone_tumor_pic")
        # bar = tqdm(pr.keys(), ncols=100)
        # for pic in bar:
        #     bar.desc = pic
        #     try:
        #         path = pic
        #         pic = pic.split('_')
        #         image = np.load(fr'/data/npz_data/bone/tumor/3d/{pic[0]}_0_{pic[1]}.npz')['image'][int(pic[2])]
        #         plt.figure(figsize=(10, 10))
        #         plt.axis('off')
        #         plt.imshow(image, cmap='gray')
        #         plt.title(f'机器预测结果: {"存在异常" if pr[path] else "正常"}', fontproperties=zhfont1)
        #         plt.savefig(fr'/data/bone_tumor_pic/{"_".join(pic[:3])}.jpg', bbox_inches='tight', pad_inches=0)
        #         plt.close()
        #     except Exception as e:
        #         print(f'{pic}: {e}')
        
        # # 不同部位
        # df = {
        #     '部位': [],
        #     '医生': [],
        #     '类型': [],
        #     'auc': [],
        #     '敏感度': [],
        #     '特异性': [],
        #     '准确率': [],
        #     'auc per case': [],
        #     '敏感度 per case': [],
        #     '特异性 per case': [],
        #     '准确率 per case': [],
        #     '单张所需时间(秒)': []
        # }
        # df = pd.DataFrame(df)
        # records = {}

        # for pr, gt, pt, tm in zip(self.pre_pr, self.pre_gt, self.pre_pt, self.pre_tm):
        #     if pt[:12] in records.keys():
        #         records[pt[:12]]['pr'].append(int(pr.item()>self.ts))
        #         records[pt[:12]]['gt'].append(int(gt.item()))
        #         records[pt[:12]]['time'] += tm
        #     else:
        #         records[pt[:12]] = {
        #             'pr': [int(pr.item()>self.ts)],
        #             'gt': [int(gt.item())],
        #             'time': tm,
        #         }

        # for patient in records.keys():
        #     pr, gt = np.array(records[patient]['pr']), np.array(records[patient]['gt'])

        #     fpr, tpr, thresholds = metrics.roc_curve(gt, pr, pos_label=1)
        #     tp = np.sum(gt+pr == 2)
        #     tn = np.sum(gt+pr == 0)
        #     fn = (np.sum(gt) - np.sum(gt+pr == 2))
        #     fp = (np.sum(pr) - np.sum(gt+pr == 2))

        #     auc = metrics.auc(fpr, tpr)*100
        #     sen = tp/(tp+fn)*100
        #     spe = tn/(tn+fp)*100
        #     acc = (tp+tn)/(tp+tn+fp+fn)*100

        #     records[patient] = {
        #             "auc": auc,
        #             "sen": sen,
        #             "spe": spe,
        #             "acc": acc,
        #             'time': records[pt[:12]]['time'] / len(pr),
        #             'pr': pr,
        #             'gt': gt
        #         }

        # for organ, vb in organs.items():
        #     print(f'\n{organ}')
        #     cnt, auc_p, sen_p, spe_p, acc_p, time_p = 0, 0, 0, 0, 0, 0
        #     pr, gt = np.array([]), np.array([])

        #     for patient in records.keys():
        #         if patient[:10] in vb:
        #             cnt += 1
        #             auc_p += records[patient]['auc']
        #             sen_p += records[patient]['sen']
        #             spe_p += records[patient]['spe']
        #             acc_p += records[patient]['acc']
        #             time_p += records[patient]['time']
        #             pr = np.append(pr, records[patient]['pr'])
        #             gt = np.append(gt, records[patient]['gt'])

        #     fpr, tpr, thresholds = metrics.roc_curve(gt, pr, pos_label=1)
        #     tp = np.sum(gt+pr == 2)
        #     tn = np.sum(gt+pr == 0)
        #     fn = (np.sum(gt) - np.sum(gt+pr == 2))
        #     fp = (np.sum(pr) - np.sum(gt+pr == 2))

        #     auc = metrics.auc(fpr, tpr)*100
        #     sen = tp/(tp+fn)*100
        #     spe = tn/(tn+fp)*100
        #     acc = (tp+tn)/(tp+tn+fp+fn)*100

        #     print(f'\t\t auc={auc:.3f} sen={sen:.3f} spe={spe:.3f} acc={acc:.3f} auc_p={auc_p/cnt:.3f} sen_p={sen_p/cnt:.3f} spe_p={spe_p/cnt:.3f} acc_p={acc_p/cnt:.3f} 单张耗时: {time_p/cnt:.3f}s')

        #     result = {
        #         '部位': organ,
        #         '类型': '/',
        #         '医生': '机器识别',
        #         'auc': round(auc, 3),
        #         '敏感度': round(sen, 3),
        #         '特异性': round(spe, 3),
        #         '准确率': round(acc, 3),
        #         'auc per case': round(auc_p/cnt, 3),
        #         '敏感度 per case': round(sen_p/cnt, 3),
        #         '特异性 per case': round(spe_p/cnt, 3),
        #         '准确率 per case': round(acc_p/cnt, 3),
        #         '单张所需时间(秒)': round(time_p/cnt, 3),
        #     }
        #     df.loc[len(df)] =  result
        # df.to_excel('result.xlsx', index=False)

        # # 各病人的数值
        # self.pre_pt = np.array([i[:12] for i in self.pre_pt])
        # pre_log = []
        # for patient in np.unique(self.pre_pt):
        #     pr = self.pre_pr[self.pre_pt == patient] > self.ts
        #     gt = self.pre_gt[self.pre_pt == patient]

        #     tp = torch.sum(gt+pr == 2)
        #     tn = torch.sum(gt+pr == 0)
        #     fn = (torch.sum(gt) - torch.sum(gt+pr == 2))
        #     fp = (torch.sum(pr) - torch.sum(gt+pr == 2))
        #     fpr, tpr, thresholds = metrics.roc_curve(gt, pr, pos_label=1)
        #     pre_log.append([patient, f'{metrics.auc(fpr, tpr)*100:.2f}', f'{tp/(tp+fn)*100:.2f}', f'{tn/(tn+fp)*100:.2f}', f'{(tp+tn)/(tp+tn+fp+fn)*100:.2f}'])
        # df = pd.DataFrame(pre_log, columns=["病人", "auc", "sen", "spe", "acc"])
        # df.to_excel(f'pred.xlsx', index=False)


    def _get_name(self):
        return self.model
