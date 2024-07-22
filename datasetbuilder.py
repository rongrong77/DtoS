import numpy as np
import torch
from torch.utils.data import Dataset
from preprocessing.transformation.transformation import Transformation
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch


class DataSetBuilder(Dataset):
    def __init__(self, x, y, labels):
        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
        self.labels = labels
        self.y_label = []
        
        self.n_sample = len(y)

        # x = np.transpose(self.x, (0, 2, 1))
        self.x = torch.from_numpy(self.x).double()
        self.y = torch.from_numpy(self.y).double().view(-1,1)

    def __len__(self):
        return self.n_sample

    def __getitem__(self, item):
        return self.x[item], self.y[item]