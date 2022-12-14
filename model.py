import os
import logging
import traceback
from collections import OrderedDict
from typing import Container
import torch.nn as nn
import torch
import torch.nn.functional as F
from test_tube import HyperOptArgumentParser
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as ptl
#from pytorch_lightning.root_module.root_module import LightningModule
from pytorch_lightning import LightningModule
from dataset import MTSFDataset
from dsanet.Layers import EncoderLayer, DecoderLayer
import argparse
from pytorch_lightning.utilities import _OMEGACONF_AVAILABLE
class Single_Global_SelfAttn_Module(nn.Module):

    def __init__(
            self,
            window, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):
        '''
        Args:

        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        '''

        super(Single_Global_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv2 = nn.Conv2d(1, n_kernels, (window, w_kernel))
        #得到n_kernals*D的数据，D为时序曲线数量
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, return_attns=False):
        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x2 = F.relu(self.conv2(x))
        #batch*n_kernel*1*D
        x2 = nn.Dropout(p=self.drop_prob)(x2)
        x = torch.squeeze(x2, 2)
        #batch*n_kernel*D
        x = torch.transpose(x, 1, 2)
        #batchsize*D*n_kernel
        src_seq = self.in_linear(x)
        #batchsize*D*d_model
        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        #batch*D*n_kernels
        return enc_output,


class Single_Local_SelfAttn_Module(nn.Module):

    def __init__(
            self,
            window, local, n_multiv, n_kernels, w_kernel,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):
        '''
        Args:

        window (int): the length of the input window size
        n_multiv (int): num of univariate time series
        n_kernels (int): the num of channels
        w_kernel (int): the default is 1
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        d_inner (int): the inner-layer dimension of Position-wise Feed-Forward Networks
        n_layers (int): num of layers in Encoder
        n_head (int): num of Multi-head
        drop_prob (float): the probability of dropout
        '''

        super(Single_Local_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(1, n_kernels, (local, w_kernel))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, return_attns=False):

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x1 = F.relu(self.conv1(x))
        x1 = self.pooling1(x1)
        x1 = nn.Dropout(p=self.drop_prob)(x1)
        x = torch.squeeze(x1, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


class AR(nn.Module):

    def __init__(self, window):

        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x


class DSANet(LightningModule):

    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        """
        super(DSANet, self).__init__()
        self.save_hyperparameters(hparams)
        self.hp = hparams

        self.batch_size = hparams.batch_size

        # parameters from dataset
        self.window = hparams.window
        self.local = hparams.local
        self.n_multiv = hparams.n_multiv
        self.n_kernels = hparams.n_kernels
        self.w_kernel = hparams.w_kernel

        # hyperparameters of model
        self.d_model = hparams.d_model
        self.d_inner = hparams.d_inner
        self.n_layers = hparams.n_layers
        self.n_head = hparams.n_head
        self.d_k = hparams.d_k
        self.d_v = hparams.d_v
        self.drop_prob = hparams.drop_prob

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        """
        self.sgsf = Single_Global_SelfAttn_Module(
            window=self.window, n_multiv=self.n_multiv, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.slsf = Single_Local_SelfAttn_Module(
            window=self.window, local=self.local, n_multiv=self.n_multiv, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.ar = AR(window=self.window)
        self.W_output1 = nn.Linear(2 * self.n_kernels, 1)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        """
        sgsf_output, *_ = self.sgsf(x)
        slsf_output, *_ = self.slsf(x)
        sf_output = torch.cat((sgsf_output, slsf_output), 2)
        sf_output = self.dropout(sf_output)
        sf_output = self.W_output1(sf_output)

        sf_output = torch.transpose(sf_output, 1, 2) 

        ar_output = self.ar(x)

        output = sf_output + ar_output

        return output

    def loss(self, labels, predictions):
        if self.hp.criterion == 'l1_loss':
            loss = F.l1_loss(predictions, labels)
        elif self.hp.criterion == 'mse_loss':
            loss = F.mse_loss(predictions, labels)
        return loss

    def training_step(self, data_batch, batch_idx):
        """
        Lightning calls this inside the training loop
        """
        # forward pass
        x, y = data_batch

        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.strategy == 'dp':
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'loss': loss_val
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, data_batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        """
        x, y = data_batch

        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.strategy == 'dp':
            loss_val = loss_val.unsqueeze(0)

        self.log("val_loss", loss_val)

        output = OrderedDict({
            'val_loss': loss_val,
            'y': y,
            'y_hat': y_hat,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        loss_sum = 0
        for x in outputs:
            loss_sum += x['val_loss'].item()
        val_loss_mean = loss_sum / len(outputs)

        y = torch.cat(([x['y'] for x in outputs]), 0)
        y_hat = torch.cat(([x['y_hat'] for x in outputs]), 0)

        num_var = y.size(-1)
        y = y.view(-1, num_var)
        y_hat = y_hat.view(-1, num_var)
        sample_num = y.size(0)

        y_diff = y_hat - y
        y_mean = torch.mean(y)
        y_translation = y - y_mean

        val_rrse = torch.sqrt(torch.sum(torch.pow(y_diff, 2))) / torch.sqrt(torch.sum(torch.pow(y_translation, 2)))
        
        y_m = torch.mean(y, 0, True)
        y_hat_m = torch.mean(y_hat, 0, True)
        y_d = y - y_m
        y_hat_d = y_hat - y_hat_m
        corr_top = torch.sum(y_d * y_hat_d, 0)
        corr_bottom = torch.sqrt( (torch.sum( torch.pow(y_d, 2), 0) * torch.sum(torch.pow(y_hat_d, 2), 0)) )
        corr_inter = corr_top / corr_bottom
        val_corr = (1./ num_var) * torch.sum(corr_inter)

        val_mae = (1./ (sample_num * num_var)) * torch.sum(torch.abs(y_diff))

        tqdm_dic = {
            'val_loss': val_loss_mean, 
            'RRSE': val_rrse.item(), 
            'CORR': val_corr.item(),
            'MAE': val_mae.item()
        }
        return tqdm_dic

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hp.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]    # It is encouraged to try more optimizers and schedulers here

    def mydataloader(self, train):
        # init data generators
        set_type = train
        dataset = MTSFDataset(window=self.hp.window, horizon=self.hp.horizon, data_name=self.hp.data_name, set_type=set_type, data_dir=self.hp.data_dir)
        # when using multi-node we need to add the datasampler
        train_sampler = None
        batch_size = self.hp.batch_size

        try:
            if self.on_gpu:
                train_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
                batch_size = batch_size // self.trainer.world_size  # scale batch size
        except Exception as e:
            pass 

        should_shuffle = train_sampler is None

        if train == 'validation':
            should_shuffle = False
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=self.hp.num_workers
        )
        return loader

    #@ptl.data_loader
    def train_dataloader(self):
        print('tng data loader called')
        return self.mydataloader(train='train')

    #@ptl.data_loader
    def val_dataloader(self):
        print('val data loader called')
        return self.mydataloader(train='validation')

    #@ptl.data_loader
    def test_dataloader(self):
        print('test data loader called')
        return self.mydataloader(train='test')

    @staticmethod
    def add_model_specific_args(): # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--local', default=3, type=int,choices=[3, 5, 7]) 
        parser.add_argument('--n_kernels', default=32, type=int, choices=[32, 50, 100])
        parser.add_argument('-w_kernel', type=int, default=1)
        parser.add_argument('--d_model', type=int, default=512, choices=[512])
        parser.add_argument('--d_inner', type=int, default=2048, choices=[2048])
        parser.add_argument('--d_k', type=int, default=64)
        parser.add_argument('--d_v', type=int, default=64)
        parser.add_argument('--n_head', type=int, default=8)
        parser.add_argument('--n_layers', type=int, default=1)
        parser.add_argument('--drop_prob', type=float, default=0.1, choices=[0.1,0.2,0.5])
    
        parser.add_argument('--data_name',  default='0', type=str)
        parser.add_argument('--data_dir', default='./AnoTransfer-data/real-world/', type=str)
        parser.add_argument('--n_multiv', type=int)
        parser.add_argument('--window', default=64, type=int, choices=[32,64,128])
        parser.add_argument('--horizon', default=3, type=int, choices=[3,6,12,24])
    
        parser.add_argument('--learning_rate', default=0.005, type=float, choices=[0.0001,0.0005,0.001,0.005,0.008])
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--criterion', default='mse_loss', type=str, choices=['l1_loss', 'mse_loss'])
    
        parser.add_argument('--batch_size', default=16, type=int, choices=[16,32,64,128,256])
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--only_test', type=int, default=0)
        return parser
