import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from engine import *
# from models.sasnet import SASNet
from models import build_model
from datasets.loading_data import SASNet_Lightning

from models.sasnet import SASNet

import os
import warnings

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training SASNet', add_help=False)
    # parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    # parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr_drop', default=300, type=int)
    parser.add_argument('--block_size', default=32, type=int)

    # parser.add_argument('--data_root', default='./ShanghaiTech',
    parser.add_argument('--data_root', default='./DATA_ROOT',
                        help='path where the dataset is')

    parser.add_argument('--checkpoints_dir', default='./weights',
                        help='path where to save checkpoints, empty for no saving')
    # parser.add_argument('--tensorboard_dir', default='./runs',
    #                     help='path where to save, empty for no saving')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--log_para', default=1000, type=int, help='scaling factor')


    return parser

def main(args):
    pl.seed_everything(42, workers=True)
    print(args)

    best_mae_checkpoint_callback = ModelCheckpoint(
        monitor='val_rmse',
        dirpath=args.checkpoints_dir,
        filename='best_rmse_model-{epoch:02d}-{val_rmse:.2f}',
        save_top_k=1,
        mode='min',
        every_n_epochs=1
    )

    latest_checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        filename='latest_model-{epoch:02d}-{val_rmse:.2f}',
    )
    
    model = build_model(args)
    
    logger = TensorBoardLogger(save_dir='./logs', name='SASNet')
    dm = SASNet_Lightning(args.data_root, args.batch_size,
                        args.num_workers, args.pin_memory)
    trainer = pl.Trainer(devices=4, accelerator="gpu", logger=logger,
                         callbacks=[best_mae_checkpoint_callback, latest_checkpoint_callback])
    trainer.fit(model, dm, ckpt_path=args.resume if args.resume else None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SASNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)