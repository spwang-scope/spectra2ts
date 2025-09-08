import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path='../dataset/ETT-small', flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scaler=None):
        
        self.args = args
        # info
        
        self.seq_len = size[0]
        self.pred_len = size[1]

        # init
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target

        self.root_path = root_path
        self.data_path = data_path
        self.scaler = scaler  # For test set, use provided scaler
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        data = df_data.values
        
        # TSLib Standard: Feature-wise StandardScaler normalization
        if self.flag == 'train' and self.scaler is None:
            # Fit scaler on training data only
            train_data = data[border1s[0]:border2s[0]]
            self.scaler = StandardScaler()
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
        elif self.flag != 'train':   # For test/valid, use the scaler found in loaded checkpoint
            if self.scaler is not None:
                data = self.scaler.transform(data)
            else:
                raise ValueError("Scaler not found.")

        
        data = torch.FloatTensor(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
    def inverse_transform(self, data):
        """TSLib standard inverse transform for evaluation if needed"""
        if self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end + 1
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
