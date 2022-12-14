import os
import logging
import traceback
from collections import OrderedDict
from turtle import position
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
from dataset import MTSFDataset,UniDataset
from dsanet.Layers import EncoderLayer, DecoderLayer
import argparse
from pytorch_lightning.utilities import _OMEGACONF_AVAILABLE
import numpy as np
class Single_Global_SelfAttn_Module(nn.Module):

    def __init__(
            self,
            window, n_multiv, n_positive, n_negative, n_random, n_kernels, w_kernel,
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
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.n_random = n_random
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv2 = nn.Conv2d(1, n_kernels, (window, w_kernel))
        #得到n_kernals*D的数据，D为时序曲线数量
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, mode, return_attns=False):
        if mode=='positive':
            self.n_multiv = self.n_positive+1
        if mode=='negetive':
            self.n_multiv = self.n_negative+1
        if mode=='random':
            self.n_random = self.n_random+1
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


class Attn_Module(nn.Module):

    def __init__(
            self,
            window, embedding_dim, local, n_multiv, n_kernels, w_kernel,
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

        super(Attn_Module, self).__init__()

        self.window = window
        self.embedding_dim = embedding_dim
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv2 = nn.Conv2d(1, n_kernels, (embedding_dim, w_kernel))
        self.conv1 = nn.Conv2d(1, n_kernels, (local, w_kernel))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv))
        self.in_linear = nn.Linear(2*n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, 2*n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, return_attns=False):
        x = x.view(-1, self.w_kernel, self.embedding_dim, self.n_multiv)
        x1 = F.relu(self.conv1(x))
        x1 = self.pooling1(x1)
        x1 = nn.Dropout(p=self.drop_prob)(x1)
        x2 = F.relu(self.conv2(x))
        x2 = nn.Dropout(p=self.drop_prob)(x2)
        x1 = torch.squeeze(x1, 2)
        x2 = torch.squeeze(x2, 2)
        #batch * kernel * D
        x = torch.cat((x1,x2),1)
        x = torch.transpose(x, 1, 2)
        #batch*D*(2*kernel)
        src_seq = self.in_linear(x)
        #batch*D*d_model
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
        #batch*1*(2*kernel) last dimenson can be changed


class Single_Local_SelfAttn_Module(nn.Module):

    def __init__(
            self,
            window, local, n_multiv, n_positive, n_negative, n_random, n_kernels, w_kernel,
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
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.n_random = n_random
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(1, n_kernels, (local, w_kernel))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x, mode, return_attns=False):
        if mode=='positive':
            self.n_multiv = self.n_positive+1
        if mode=='negetive':
            self.n_multiv = self.n_negative+1
        if mode=='random':
            self.n_random = self.n_random+1
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

class AR2(nn.Module):

    def __init__(self, window1,window2):

        super(AR2, self).__init__()
        self.linear1 = nn.Linear(window1, 1)
        self.linear2 = nn.Linear(window2, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear1(x)
        x = torch.squeeze(x, 2)
        x = self.linear2(x)
        return x

class DSANet(LightningModule):

    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        """
        super(DSANet, self).__init__()
        self.save_hyperparameters()
        self.hp = hparams
        print(hparams)

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
        
        #new
        self.n_positive = hparams.n_positive
        self.n_negative = hparams.n_negative
        self.n_random = hparams.n_random

        self.embedding_dim = hparams.embedding_dim

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        """
        """ self.sgsf = Single_Global_SelfAttn_Module(
            window=self.window, n_multiv=self.n_multiv, n_positeve = self.n_positive, negative = self.n_negative, random = self.n_random, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.slsf = Single_Local_SelfAttn_Module(
            window=self.window, local=self.local, n_multiv=self.n_multiv,  n_positeve = self.n_positive, negative = self.n_negative, random = self.n_random, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob) """
        
        self.attn1 = Attn_Module(
            window=self.window, embedding_dim=self.embedding_dim, local=self.local, n_multiv=2*self.n_positive+1,  n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)
        self.attn2 = Attn_Module(
            window=self.window, embedding_dim=self.embedding_dim, local=self.local, n_multiv=self.n_negative+1, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)
        self.attn3 = Attn_Module(
            window=self.window, embedding_dim=self.embedding_dim, local=self.local, n_multiv=self.n_random+1,  n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)
        self.attn4 = Attn_Module(
            window=self.window, embedding_dim=self.embedding_dim, local=self.local, n_multiv=self.n_random+1,  n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)
        self.attn5 = Attn_Module(
            window=self.window, embedding_dim=self.embedding_dim, local=self.local, n_multiv=self.n_random+1, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.ar = AR2(window1=self.embedding_dim,window2=1+2*self.n_positive+self.n_negative+self.n_random)
        self.W_output1 = nn.Linear(10 * self.n_kernels, 1)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()
        self.cross_linear1 = nn.Linear(2*self.n_kernels, self.embedding_dim)
        self.cross_linear2 = nn.Linear(2*self.n_kernels, self.embedding_dim)
        
        self.embedding = nn.Embedding(self.window, self.embedding_dim)
        self.embedding2 = nn.Linear(self.window, self.embedding_dim)


    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        """
        #batch*w*D
        #BATCH*D*W =>batch*D*W*embedding_dim  =>batch*D*embeddingdim
        pos = torch.arange(0,self.window)
        pos = pos.view(-1,self.window)
        x_emb1 = torch.mean(self.embedding(pos),dim=1)#1*embeding_dim
        batch, _, dim = x.size()
        x_emb1 = x_emb1.unsqueeze(0)
        x_emb1 = torch.transpose(x_emb1.repeat(batch, dim, 1),1,2)
        #batch*embding_dim*D

                

        x_emb2 = torch.transpose(self.embedding2(torch.transpose(x,1,2)),1,2)
        #batch*embeding_dim*D
        x = x_emb1+x_emb2


        x_target = x[:,:,0]
        x_target = x_target.unsqueeze(2)
        x_positive = x[:,:,1:1+2*self.n_positive]
        x_random = x[:,:,1+2*self.n_positive:1+2*self.n_positive+self.n_random]
        x_negative = x[:,:,1+2*self.n_positive+self.n_random:1+2*self.n_positive+self.n_random+self.n_negative]


        attn1_out, *_ = self.attn1(torch.cat((x_positive,x_target),2))
        attn2_out, *_ = self.attn2(torch.cat((x_negative,x_target),2))
        attn3_out, *_ = self.attn3(torch.cat((x_random,x_target),2))
        #batch*1*(2*kernel)
        '''vallina attention'''
        cross_positive = self.cross_linear1(attn1_out)
        cross_positive = torch.transpose(cross_positive,1,2)
        cross_negative = self.cross_linear2(attn2_out)
        cross_negative = torch.transpose(cross_negative,1,2)
        #batch*w*1

        attn4_out, *_ = self.attn4(torch.cat((x_random,cross_positive),2))
        attn5_out, *_ = self.attn5(torch.cat((x_random,cross_negative),2))

        """ sgsf_output_positive, *_ = self.sgsf(torch.cat((x_positive,x_target),2),mode='positive')
        slsf_output_positive, *_ = self.slsf(torch.cat((x_positive,x_target),2),mode='positive')

        sgsf_output_negative, *_ = self.sgsf(torch.cat((x_negative,x_target),2),mode='negative')
        slsf_output_negative, *_ = self.slsf(torch.cat((x_negative,x_target),2),mode='negative')

        sgsf_output_random, *_ = self.sgsf(torch.cat((x_random,x_target),2),mode='random')
        slsf_output_random, *_ = self.slsf(torch.cat((x_random,x_target),2),mode='random')
        #batch*1*n_kernel

        sf_output_positive = torch.cat((sgsf_output_positive, slsf_output_positive), 2)
        sf_output_negative = torch.cat((sgsf_output_negative, slsf_output_negative), 2)
        sf_output_random = torch.cat((sgsf_output_random, slsf_output_random), 2)
        #batch*1*(2*n_kernel)
        '''vallina attention'''
        cross_positive = self.cross_linear(sf_output_positive)
        cross_positive = torch.transpose(cross_positive,1,2)
        cross_negative = self.cross_linear(sf_output_negative)
        cross_negative = torch.transpose(cross_negative,1,2)

        sgsf_output_cross_positive, *_ = self.sgsf(torch.cat((x_random,cross_positive),2),mode='random')
        slsf_output_cross_negative, *_ = self.slsf(torch.cat((x_random,cross_negative),2),mode='random')

        sgsf_output_random, *_ = self.sgsf(torch.cat((x_random,x_target),2),mode='random')
        slsf_output_random, *_ = self.slsf(torch.cat((x_random,x_target),2),mode='random')
        """
        out = torch.cat((attn1_out,attn2_out,attn3_out,attn4_out,attn5_out),2)
        out = torch.squeeze(out, 1)
        out = self.dropout(out)
        out = self.W_output1(out)
        #batch*1

        ar_output = self.ar(x)

        output = out + ar_output

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

        self.log("val_loss_train", loss_val, on_step=True, on_epoch=True, logger= True)

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

        self.log("val_loss_valid", loss_val, on_step=True, on_epoch=True, logger= True)

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
    
    def test_step(self, data_batch, batch_idx):
        x , y = data_batch
        y_hat = self.forward(x)
        loss_val = self.loss(y, y_hat)
        self.log("val_loss_test", loss_val, on_step=True, on_epoch=True, logger= True)
        output = OrderedDict({
            'val_loss': loss_val,
            'y': y,
            'y_hat': y_hat,
        })
        
        return output

    def test_epoch_end(self, outputs):
        y = torch.cat(([x['y'] for x in outputs]), 0)
        y_hat = torch.cat(([x['y_hat'] for x in outputs]), 0)
        #loss = F.mse_loss(y,y_hat)
        loss = y - y_hat
        loss = torch.pow(loss, 2)
        y_mean = torch.mean(y)
        y = y - y_mean.item()
        y = torch.pow(y, 2)
        print(loss)
        print(y)
        rrse = torch.sqrt(torch.sum(loss)/torch.sum(y))
        print("rrse",rrse)
        mse = torch.mean(loss)
        print("mse",mse)

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
        dataset = UniDataset(window=self.hp.window, horizon=self.hp.horizon, n_positive = self.hp.n_positive, n_negative = self.hp.n_negative, n_random = self.hp.n_random, data_name=self.hp.data_name, set_type=set_type, data_dir=self.hp.data_dir)
        #fortest
        """ x = dataset.__getitem__(1024)
        for i in range(1+2*self.n_positive+self.n_random+self.n_negative):
            print(x[0][:,i])
        print("gg")
        x= dataset.__getitem__(0)
        for i in range(1+2*self.n_positive+self.n_random+self.n_negative):
            print(x[0][:,i]) """
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

        if train == 'validation' or train == 'test':
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
        parser.add_argument('--data_dir', default='./AnoTransfer-data/real-world-data/', type=str)
        parser.add_argument('--n_multiv', type=int,default=6)
        parser.add_argument('--window', default=64, type=int, choices=[32,64,128])
        parser.add_argument('--horizon', default=3, type=int, choices=[3,6,12,24])
    
        parser.add_argument('--learning_rate', default=0.005, type=float, choices=[0.0001,0.0005,0.001,0.005,0.008])
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--criterion', default='mse_loss', type=str, choices=['l1_loss', 'mse_loss'])
    
        parser.add_argument('--batch_size', default=16, type=int, choices=[16,32,64,128,256])
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--only_test', type=int, default=0)
        parser.add_argument('--ckpt_path', type=str, default='./ckpt/epoch=8-step=15354.ckpt')
        parser.add_argument('--hp_path', type=str, default='./logs/version_0/hparams.yaml')
        parser.add_argument('--n_positive', type=int, default=10)
        parser.add_argument('--n_negative', type=int, default=10)
        parser.add_argument('--n_random', type=int, default=10)
        parser.add_argument('--embedding_dim', type=int, default=128)
        parser.add_argument('--max_epoch', type=int, default=10)
        return parser
