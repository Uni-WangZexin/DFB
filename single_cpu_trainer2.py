"""
Runs a model on a single node on CPU only.
"""
from ast import arg
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
#only mac
import logging
import numpy as np
import torch
import traceback

#from test_tube import HyperOptArgumentParser, Experiment
#from pytorch_lightning.models.trainer import Trainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from model_self_attn2 import DSANet

from pytorch_lightning.loggers import TensorBoardLogger
import argparse

SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

logger = TensorBoardLogger(name='logs',save_dir='./')

def main(hparams):
    """
    Main training routine specific for this project
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    print('loading model...')
    model = DSANet(hparams)
    print('model built')


    early_stop = EarlyStopping(
        monitor='val_loss_valid_epoch',
        patience=5,
        verbose=True,
        mode='min'
    )
    checkpoint = ModelCheckpoint(dirpath='./ckpt/',filename='{}'.format(hparams.data_name), monitor='val_loss_valid_epoch',mode='min')
    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        max_epochs=hparams.max_epoch,
        callbacks=[early_stop,checkpoint],
        logger=logger,
        accelerator="gpu",
        devices=1,
        #strategy=
    ) 
    # ------------------------
    # 5 START TRAINING
    # ------------------------
    print("fit start")
    train_loader = model.train_dataloader()
    val_loader = model.val_dataloader()
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)
    trainer.test(model, dataloaders=model.test_dataloader())

    print('View tensorboard logs by running\ntensorboard --logdir %s' % os.getcwd())
    print('and going to http://localhost:6006 on your browser')


if __name__ == '__main__':
    parser = DSANet.add_model_specific_args()
    hyperparams = parser.parse_args()
    # ---------------------
    # RUN TRAINING
    # ---------------------
    # run on HPC cluster
    print(f'RUNNING ON CPU')
    # * change the following code to comments for grid search
    if hyperparams.only_test ==1:
        model = DSANet.load_from_checkpoint(checkpoint_path=hyperparams.ckpt_path)
        trainer = Trainer(accelerator='gpu',devices=1)
        trainer.test(model,dataloaders=model.test_dataloader())
    else:
        main(hyperparams)


    # * recover the following code for grid search
    # hyperparams.optimize_parallel_cpu(
    #     main,
    #     nb_trials=24,    # this number needs to be adjusted according to the actual situation
    #     nb_workers=1
    # )
    