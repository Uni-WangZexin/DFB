import os
import torch
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from test_tube import Experiment


""" exp = Experiment(
        name='dsanet_exp__window=_horizon=',
        save_dir='test',
        autosave=False,
        description='test demo'
    )
print(exp) """
class MTSFDataset(torch.utils.data.Dataset):
    """Multi-variate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    def __init__(self,
                window=32,
                horizon=3,
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
        print("init")
        self.samples, self.labels = self.__getsamples(rawdata)
        print("init")
    def __getsamples(self, data):
        print("sample")
        print(self.sample_num, self.window, self.var_num)
        X = torch.zeros([self.sample_num, self.window, self.var_num])
        print("sample")
        Y = torch.zeros([self.sample_num, 1, self.var_num])
        print("sample")
        for i in range(self.sample_num):
            #print(i)
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

if __name__ == "__main__":
    a= np.asarray([5,1])
    b=a[np.where(a>6)]
    print(b)
    if(b.size==0):
        print(111)