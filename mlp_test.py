# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 12:41:58 2022

@author: snourashrafeddin
"""


import numpy as np
import pandas as pd
import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold

y_col = 'target'
X_cols = ['investment_id'] + [f'f_{i}' for i in range(n_features)]
test_df = pd.read_csv('example_test.csv')

class ubiquant(nn.Module):
    def __init__(self):
        super(ubiquant, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(len(X_cols), 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
       
    def forward(self, x):
         return self.layers(x)
     
data = torch.from_numpy(test_df[X_cols].values.astype(np.float32))
preds = np.array([0]*len(test_df)).astype(np.float32)
for i in range(3):
    model = torch.load(f'model_{i}.pth')
    model.eval()
    preds += model(data).detach().numpy()[:, 0]
    

    