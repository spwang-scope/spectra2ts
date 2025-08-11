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
                 target='OT'):
        
        self.args = args
        # info
        
        self.context_length = size[0]
        self.prediction_length = size[1]

        # init
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target

        self.root_path = root_path
        self.data_path = data_path
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
        num_train = int(len(df_raw) * 0.8)
        num_test = len(df_raw) - num_train
        border1s = [0, len(df_raw) - num_test - self.context_length]
        border2s = [num_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        data = df_data.values
        
        # Fix data leakage: Only fit scaler on training data
        if self.set_type == 0:  # train
            self.normalizer = StandardScaler()
            train_data = data[border1s[0]:border2s[0]]  # Training data only
            self.normalizer.fit(train_data)
            data = self.normalizer.transform(data)
        else:  # test
            # For test set, we need to use the same scaler fitted on training data
            # This requires the scaler to be passed from training dataset
            # For now, fit on training portion of current data
            train_data = data[border1s[0]:border2s[0]]  # Training data only
            normalizer = StandardScaler()
            normalizer.fit(train_data)
            data = normalizer.transform(data)
            
        data = torch.FloatTensor(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.context_length
        r_begin = s_begin + 1
        r_end = r_begin + self.prediction_length

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.context_length - self.prediction_length + 1
