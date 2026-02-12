import argparse
import os
import torch
from pl_data import MyDataModule
from pl_model import CT_Classification as m
import lightning.pytorch as pl
import tensorboard
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
import kornia.augmentation as K
from typing import Union
from pytorch_lightning.loggers import TensorBoardLogger
from util import batch_sizes
import shutil
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
torch.set_float32_matmul_precision('high')
os.system("cls")


def main(args):

    if args.device in ['0', '1', '2', '3']:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    else:
        raise ValueError("Invalid Device")
    model_name = args.model.lower().replace("-", "_")
    save_dir = f"inf/{model_name}"

    if args.is_train:
        no = 0
        if os.path.exists(save_dir):
            while os.path.exists(fr'{save_dir}/{str(no).zfill(2)}'):
                no += 1

    else:
        if os.path.exists(str(args.weights)):
            no = int(os.path.basename(args.weights)[:2])
        else:
            raise KeyError('Predict Weight doesn\'t exist')
    save_dir = f"{save_dir}/{str(no).zfill(2)}"

    transforms = {
        'train': K.AugmentationSequential(
            K.RandomEqualize(p=0.7),
            K.RandomMotionBlur(3, 35., 0.5, p=.4),
            K.RandomSharpness(1., p=.4),
            K.RandomClahe(p=.7),
            K.RandomAffine(degrees=(-30., 30.), p=0.7),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomCrop(size=(args.pic_size, args.pic_size),
                         p=0.7, padding=30, cropping_mode='resample'),
            data_keys=["input"],
            same_on_batch=False,
        ),
        'val': K.AugmentationSequential(
            data_keys=["input"],
            same_on_batch=False,
        ),
        'predict': K.AugmentationSequential(
            data_keys=['input'],
            same_on_batch=False
        )}

    dm_paras = {
        'path': args.data_path,
        'batch_size': batch_sizes[model_name],
        'num_workers': 16,
    }
    dm = MyDataModule(**dm_paras)
    model_paras = {
        'model': model_name,
        'num_classes': 2,
        'transform': transforms,
        'lr_max': 1e-4
    }
    model = m(**model_paras)

    process_callback = RichProgressBar(
        theme=RichProgressBarTheme(
            description='green_yellow',
            progress_bar='green1',
            progress_bar_finished='blue1',
            progress_bar_pulse='red1',
            batch_progress='green_yellow',
            time='grey82',
            processing_speed='grey30',
            metrics='grey82'
        )
    )

    if args.is_train:
        dm.setup("fit")
        if args.weights != "":
            try:
                model = model.load_from_checkpoint(args.weights, **model_paras)
            except:
                raise("Weights cannot be loaded")

        checkpoint_callback_auc = ModelCheckpoint(monitor="matrix/auc",
                                                  dirpath=save_dir,
                                                  filename=f'{str(no).zfill(2)}_'+'auc_ts={threshold:.2f}-auc={matrix/auc:.2f}-acc={matrix/acc:.2f}-sen={matrix/sensitivity:.2f}-spe={matrix/specificity:.2f}',
                                                  save_last=False,
                                                  mode="max",
                                                  auto_insert_metric_name=False)
        checkpoint_callback_spe = ModelCheckpoint(monitor="matrix/sensitivity",
                                                  dirpath=save_dir,
                                                  filename=f'{str(no).zfill(2)}_'+'spe_ts={threshold:.2f}-auc={matrix/auc:.2f}-acc={matrix/acc:.2f}-sen={matrix/sensitivity:.2f}-spe={matrix/specificity:.2f}',
                                                  save_last=False,
                                                  mode="max",
                                                  auto_insert_metric_name=False)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        earlystop_callback = EarlyStopping(
            monitor="save/check", mode="max", patience=5)
        logger = TensorBoardLogger(
            save_dir=save_dir,
            name=f'',
            version=f'log',
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs, check_val_every_n_epoch=1,
            min_epochs=20,
            log_every_n_steps=1,
            logger=logger,
            callbacks=[checkpoint_callback_auc, checkpoint_callback_spe,
                       process_callback, lr_monitor, earlystop_callback],
            reload_dataloaders_every_n_epochs=1,
            num_sanity_val_steps=0,
        )

        trainer.fit(model, datamodule=dm)
        with open(f"{save_dir}/val_patients.txt", 'w') as f:
            f.write(dm.val_patients())

        value = 0
        for k, v in checkpoint_callback_auc.best_k_models.items():
            if v > value:
                value = v
                args.weights = k

    dm.setup('predict')
    if args.weights != "":
        try:
            ts = float(re.findall(r'_ts=(.{4})-', args.weights)[0])
            model = m.load_from_checkpoint(args.weights, ts=ts, **model_paras)
            model.eval()
        except:
            raise KeyError("No such weight")
    else:
        raise KeyError("Missing Weight Path")
    model.check_gradecam(dataloader=dm.predict_dataloader())
    # trainer = pl.Trainer(
    #     # enable_progress_bar=False,
    #     enable_checkpointing=False,
    #     enable_model_summary=False,
    #     logger=False,
    #     )
    # trainer.predict(model, datamodule=dm)

    # if os.path.exists('pred.xlsx'):
    #     os.rename('pred.xlsx', f'{save_dir}/pred.xlsx')

    # if os.path.exists('result.xlsx'):
    #     os.rename('result.xlsx', f'{save_dir}/result.xlsx')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    # /data/npz_data/bone/inf/slice/ct_tra   /data/npz_data/bone/tumor/slice/ct_tra
    parser.add_argument(
        '--data_path', type=Union[str, list], default=[r'E:\project\bone\data\npz'])
    parser.add_argument('--weights', type=str, help='',
                        default=r'E:\project\bone\pre_classification\inf\swin_base\00\00_auc_ts=0.54-auc=95.59-acc=87.52-sen=90.30-spe=87.18.ckpt', )
    parser.add_argument('--device', default='0',
                        help='device id (i.e. 0 or 1 or 2 or 3)')
    parser.add_argument('--is_train', type=bool, default=False)
    parser.add_argument('--pic_size', type=int, default=224)
    parser.add_argument('--time', type=int, default=None)
    parser.add_argument('--model', type=str, default='swin_base')

    opt = parser.parse_args()

    main(opt)
