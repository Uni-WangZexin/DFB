
from cProfile import label
from cmath import pi
from curses import window
import os
from turtle import position
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.utils.data

import logging
import numpy as np
import pandas as pd
import datapreprocess

class MTSFDataset(torch.utils.data.Dataset):
    """Multi-variate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    def __init__(self,
                window,
                horizon,
                data_name='electricity',
                set_type='train',    # 'train'/'validation'/'test'
                data_dir='./data/multivariate-time-series-data/'):
        assert type(set_type) == type('str')
        self.window = window
        self.horizon = horizon
        self.data_dir = data_dir
        self.set_type = set_type

        file_path = os.path.join(data_dir, data_name, '{}_{}.txt'.format(data_name, set_type))


        rawdata = np.loadtxt(open(file_path), delimiter=',')
        self.len, self.var_num = rawdata.shape
        self.sample_num = max(self.len - self.window - self.horizon + 1, 0)
        self.samples, self.labels = self.__getsamples(rawdata)

    def __getsamples(self, data):
        print(self.sample_num, self.window, self.var_num)
        X = torch.zeros([self.sample_num, self.window, self.var_num])
        Y = torch.zeros([self.sample_num, 1, self.var_num])
        for i in range(self.sample_num):
            start = i
            end = i + self.window
            X[i, :, :] = torch.from_numpy(data[start:end, :])
            Y[i, :, :] = torch.from_numpy(data[end+self.horizon-1, :])
        return (X, Y)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :, :]]
        return sample


class UniDataset(torch.utils.data.Dataset):
    """Multi-variate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    """ def __init__(self,
                window,
                horizon,
                data_name='wecar',
                set_type='train',    # 'train'/'validation'/'test'
                data_dir='./data'):
        assert type(set_type) == type('str')
        self.window = window
        self.horizon = horizon
        self.data_dir = data_dir
        self.set_type = set_type

        file_path = os.path.join(data_dir, data_name, '{}_{}.txt'.format(data_name, set_type))

        rawdata = np.loadtxt(open(file_path), delimiter=',')
        self.len, self.var_num = rawdata.shape
        self.sample_num = max(self.len - self.window - self.horizon + 1, 0)
        self.samples, self.labels = self.__getsamples(rawdata) """
    
    def __init__(self,
            window,
            horizon,
            n_positive = 5,
            n_negative =5,
            n_random = 5,
            data_name = '0',
            set_type = 'train',
            data_dir ='./AnoTransfer-data/real-world-data-standard/'):
        self.window = window
        self.horizon = horizon
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.n_random = n_random
        self.data_dir = data_dir
        self.set_type = set_type
        self.data_name = data_name
        file_path = os.path.join(data_dir, data_name, '{}_{}.csv'.format(data_name, set_type))
        df = pd.read_csv(file_path)
        timestamp, values, labels, missing = np.asarray(df['timestamp']), np.asarray(df['value']), np.asarray(df['label']), np.asarray(df['missing'])
        #labels = np.zeros_like(values, dtype=np.int32)
        #timestamp, missing, (values, labels) = datapreprocess.complete_timestamp(timestamp,(values,labels))
        exclude = np.logical_or(labels.astype(int), missing)
        include_index = np.where(exclude==0)[0]
        include_index = include_index[np.where(include_index>=(window+horizon-1))]
        self.len = len(values)
        print("sss",len(values))
        self.sample_num = max(self.len - window - horizon + 1, 0)
        self.samples, self.labels = self.__getsamples(values,labels,timestamp,include_index)


    """ def __getsamples(self, data):
        X = torch.zeros((self.sample_num, self.window, self.var_num))
        Y = torch.zeros((self.sample_num, 1, self.var_num))

        for i in range(self.sample_num):
            start = i
            end = i + self.window
            X[i, :, :] = torch.from_numpy(data[start:end, :])
            Y[i, :, :] = torch.from_numpy(data[end+self.horizon-1, :])

        return (X, Y) """
    
    def __getsamples(self, data,labels,timestamp,include_index):
        anomaly_index_all = np.where(labels == 1)[0]
        X = torch.zeros((self.sample_num, self.window+1, 1+2*self.n_positive+self.n_negative+self.n_random))
        Y = torch.zeros((self.sample_num, 1))
        time_interval = timestamp[1] - timestamp[0]
        position = np.arange(start = 1,stop = self.n_positive+1,dtype=np.int32)
        position1 = position*24*60*60/time_interval
        position2 = position*60*60/time_interval
        for i in range(self.sample_num):
            """ if(self.set_type=='train):
                if(labels[i+self.window+self.horizon-1]==0):
                    Y[i,:] =data[i+self.window+self.horizon-1]
                else:
                    now = i+self.window+self.horizon-1
                    norm = 0
                    for before in range(now,0,-1):
                        if(labels[before]==0):
                            norm = before
                            break
                    Y[i,:] = data[norm]
            else: """
            Y[i,:] =data[i+self.window+self.horizon-1]
            X[i,0:self.window,0] = torch.from_numpy(data[i:i+self.window])
            positive1 = i - position1
            #positive1 = positive1[np.where((positive1)>=0)]
            positive1 = np.sort(positive1)    
            for j in range(self.n_positive):
                start = int(positive1[j])
                if(start<0):
                    continue
                end = start+self.window
                X[i,0:self.window,j+1] = torch.from_numpy(data[start:end])
                X[i,self.window,j+1] = torch.from_numpy(np.asarray(data[end+self.horizon-1]))
            positive2 = i - position2
            #positive2 = positive2[np.where((positive2)>=0)]
            positive2 = np.sort(positive2)
            for j in range(self.n_positive):
                start = int(positive2[j])
                if(start<0):
                    continue
                end = start+self.window
                X[i,0:self.window,j+self.n_positive+1] = torch.from_numpy(data[start:end])
                X[i,self.window,j+self.n_positive+1] = torch.from_numpy(np.asarray(data[end+self.horizon-1]))
            rand = np.random.randint(low=0,high=i+1,size=self.n_random,dtype=np.int32)
            rand = np.sort(rand)
            for j in range(self.n_random):
                start = rand[j]
                end = rand[j] + self.window
                X[i,0:self.window,j+2*self.n_positive+1] = torch.from_numpy(data[start:end])
                X[i,self.window,j+2*self.n_positive+1] = torch.from_numpy(np.asarray(data[end+self.horizon-1]))
            anomaly_index = anomaly_index_all[np.where(anomaly_index_all<(i+self.window-1))]
            anomaly_index = anomaly_index[np.where((anomaly_index-self.window-self.horizon+1)>0)]
            if(anomaly_index.size == 0):
                continue
            rand_ano = np.random.choice(anomaly_index,size = self.n_negative, replace=True)
            rand_ano = np.sort(rand_ano)
            for j in range(self.n_negative):
                start = rand_ano[j]-self.window-self.horizon+1
                end = rand_ano[j]-self.horizon+1
                X[i, 0:self.window ,j+2*self.n_positive+self.n_random+1] = torch.from_numpy(data[start:end])
                X[i, self.window ,j+2*self.n_positive+self.n_random+1] = torch.from_numpy(np.asarray(data[rand_ano[j]]))
        if(self.set_type=='train'):
            X = torch.index_select(X, dim=0, index = torch.from_numpy(include_index-self.window-self.horizon+1))
            Y = torch.index_select(Y, dim=0, index = torch.from_numpy(include_index-self.window-self.horizon+1))
            self.sample_num = X.shape[0]
        return (X,Y)


    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :]]
        return sample




class UniDataset2(torch.utils.data.Dataset):
    """Multi-variate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    """ def __init__(self,
                window,
                horizon,
                data_name='wecar',
                set_type='train',    # 'train'/'validation'/'test'
                data_dir='./data'):
        assert type(set_type) == type('str')
        self.window = window
        self.horizon = horizon
        self.data_dir = data_dir
        self.set_type = set_type

        file_path = os.path.join(data_dir, data_name, '{}_{}.txt'.format(data_name, set_type))

        rawdata = np.loadtxt(open(file_path), delimiter=',')
        self.len, self.var_num = rawdata.shape
        self.sample_num = max(self.len - self.window - self.horizon + 1, 0)
        self.samples, self.labels = self.__getsamples(rawdata) """
    
    def __init__(self,
            window,
            horizon,
            n_positive = 5,
            n_negative =5,
            n_random = 5,
            data_name = '0',
            set_type = 'train',
            data_dir ='./AnoTransfer-data/real-world-data-standard/'):
        self.window = window
        self.horizon = horizon
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.n_random = n_random
        self.data_dir = data_dir
        self.set_type = set_type
        self.data_name = data_name
        file_path = os.path.join(data_dir, data_name, '{}_{}.csv'.format(data_name, set_type))
        df = pd.read_csv(file_path)
        timestamp, values, labels, missing = np.asarray(df['timestamp']), np.asarray(df['value']), np.asarray(df['label']), np.asarray(df['missing'])
        #labels = np.zeros_like(values, dtype=np.int32)
        #timestamp, missing, (values, labels) = datapreprocess.complete_timestamp(timestamp,(values,labels))
        exclude = np.logical_or(labels.astype(int), missing)
        include_index = np.where(exclude==0)[0]
        include_index = include_index[np.where(include_index>=(window+horizon-1))]
        self.len = len(values)
        print("sss",len(values))
        self.sample_num = max(self.len - window - horizon + 1, 0)
        self.samples, self.labels = self.__getsamples(values,labels,timestamp,include_index)


    """ def __getsamples(self, data):
        X = torch.zeros((self.sample_num, self.window, self.var_num))
        Y = torch.zeros((self.sample_num, 1, self.var_num))

        for i in range(self.sample_num):
            start = i
            end = i + self.window
            X[i, :, :] = torch.from_numpy(data[start:end, :])
            Y[i, :, :] = torch.from_numpy(data[end+self.horizon-1, :])

        return (X, Y) """
    
    def __getsamples(self, data,labels,timestamp,include_index):
        anomaly_index_all = np.where(labels == 1)[0]
        X = torch.zeros((self.sample_num, self.window+1, 1+2*self.n_positive+self.n_negative+self.n_random))
        Y = torch.zeros((self.sample_num, 1))
        time_interval = timestamp[1] - timestamp[0]
        position = np.arange(start = 1,stop = self.n_positive+1,dtype=np.int32)
        position1 = position*24*60*60/time_interval
        position2 = position*60*60/time_interval
        for i in range(self.sample_num):
            """ if(self.set_type=='train):
                if(labels[i+self.window+self.horizon-1]==0):
                    Y[i,:] =data[i+self.window+self.horizon-1]
                else:
                    now = i+self.window+self.horizon-1
                    norm = 0
                    for before in range(now,0,-1):
                        if(labels[before]==0):
                            norm = before
                            break
                    Y[i,:] = data[norm]
            else: """
            Y[i,:] =data[i+self.window+self.horizon-1]
            X[i,0:self.window,0] = torch.from_numpy(data[i:i+self.window])
            positive1 = i - position1
            #positive1 = positive1[np.where((positive1)>=0)]
            positive1 = np.sort(positive1)    
            for j in range(self.n_positive):
                start = int(positive1[j])
                if(start<0):
                    continue
                end = start+self.window
                X[i,0:self.window,j+1] = torch.from_numpy(data[start:end])
                X[i,self.window,j+1] = torch.from_numpy(np.asarray(data[end+self.horizon-1]))
            positive2 = i - position2
            #positive2 = positive2[np.where((positive2)>=0)]
            positive2 = np.sort(positive2)
            for j in range(self.n_positive):
                start = int(positive2[j])
                if(start<0):
                    continue
                end = start+self.window
                X[i,0:self.window,j+self.n_positive+1] = torch.from_numpy(data[start:end])
                X[i,self.window,j+self.n_positive+1] = torch.from_numpy(np.asarray(data[end+self.horizon-1]))
            rand = np.random.randint(low=0,high=i+1,size=self.n_random,dtype=np.int32)
            rand = np.sort(rand)
            for j in range(self.n_random):
                start = rand[j]
                end = rand[j] + self.window
                X[i,0:self.window,j+2*self.n_positive+1] = torch.from_numpy(data[start:end])
                X[i,self.window,j+2*self.n_positive+1] = torch.from_numpy(np.asarray(data[end+self.horizon-1]))
            anomaly_index = anomaly_index_all[np.where(anomaly_index_all<(i+self.window-1))]
            anomaly_index = anomaly_index[np.where((anomaly_index-self.window-self.horizon+1)>0)]
            if(anomaly_index.size == 0):
                continue
            rand_ano = np.random.choice(anomaly_index,size = self.n_negative, replace=True)
            rand_ano = np.sort(rand_ano)
            for j in range(self.n_negative):
                start = rand_ano[j]-self.window-self.horizon+1
                end = rand_ano[j]-self.horizon+1
                X[i, 0:self.window ,j+2*self.n_positive+self.n_random+1] = torch.from_numpy(data[start:end])
                X[i, self.window ,j+2*self.n_positive+self.n_random+1] = torch.from_numpy(np.asarray(data[rand_ano[j]]))
        if(set_type=='train'):
            X = torch.index_select(X, dim=0, index = torch.from_numpy(include_index-self.window-self.horizon+1))
            Y = torch.index_select(Y, dim=0, index = torch.from_numpy(include_index-self.window-self.horizon+1))
            self.sample_num = X.shape[0].item()
        return (X,Y)


    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :]]
        return sample
# test
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    logging.debug('Test: data from .txt file')
    sample = UniDataset(64,3,5,5,5)
    
    
