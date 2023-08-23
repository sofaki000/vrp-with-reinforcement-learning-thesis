import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import time
from IPython.display import clear_output
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TSPDataset(Dataset):

    def __init__(self, num_nodes, num_samples, random_seed=111):
        super(TSPDataset, self).__init__()
        torch.manual_seed(random_seed)

        self.data_set = []
        for l in tqdm(range(num_samples)):
            x = torch.FloatTensor(2, num_nodes).uniform_(0, 1)
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]

class TSPDatasetGaussian(Dataset):
    def __init__(self, num_nodes, num_samples, random_seed=111):
        super(TSPDatasetGaussian, self).__init__()
        torch.manual_seed(random_seed)

        self.data_set = []
        for l in tqdm(range(num_samples)):
            m = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
            x = m.sample(sample_shape=(2,num_nodes))
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]

train_size = 12
val_size = 12
train_20_dataset = TSPDataset(20, train_size)
val_20_dataset   = TSPDataset(20, val_size)
