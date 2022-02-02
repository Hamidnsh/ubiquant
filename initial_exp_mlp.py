# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:02:02 2022

@author: snourashrafeddin
"""
import numpy as np
import pandas as pd
import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import math

_seed = 2022
n_features = 300
n_splits = 3
batch_size = 1024
n_epochs = 20
patience = 3
torch.manual_seed(_seed)

y_col = 'target'
X_cols = ['investment_id'] + [f'f_{i}' for i in range(n_features)]


class ubiquant(nn.Module):
    def __init__(self):
        super(ubiquant, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(len(X_cols), 64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(32, 1)
        )
       
    def forward(self, x):
         return self.layers(x)


class custom_dataset(Dataset):
    def __init__(self, data):
        self.data = data 
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data.iloc[idx][X_cols].values.astype(np.float32)), torch.tensor(self.data.iloc[idx][y_col], dtype=torch.float)


def create_dataloader(train_df, batch_size, fold=0):
    train_set, val_set = train_df.loc[train_df['kfold'] != fold],  train_df[train_df['kfold'] == fold]
    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    train_dataset = custom_dataset(train_set)
    val_dataset = custom_dataset(val_set)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    
    return train_dataloader, val_dataloader


def load_CV_data():
    train_df = pd.read_parquet('train_low_mem.parquet')
    train_df["kfold"] = -1
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=_seed)
    for f, (t_, v_) in enumerate(kf.split(X=train_df)):
        train_df.loc[v_, 'kfold'] = f
    return train_df

def create_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ubiquant().to(device)        
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    return model, loss_fn, optimizer, device, scheduler

def train_model(train_dataloader, model, loss_fn, device, optimizer, scheduler):
    model.train()
    train_loss = 0
    size = 0
    
    for batch, (data, target) in enumerate(train_dataloader):
        data = data.to(device)
        y = target.to(device)
        
        pred = model(data)[:, 0]
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += np.sum((pred.detach().numpy() - y.detach().numpy())**2)
        size += len(y)
        
    scheduler.step()
    train_rmse = np.sqrt(train_loss/size)
    print(f'train rmse: {train_rmse} ')
    return train_rmse, model

def evaluate_model(val_dataloader, model, loss_fn, device):
    model.eval()
    test_loss = 0
    size = 0
    with torch.no_grad():
        for batch, (data, target) in enumerate(val_dataloader):
            data = data.to(device)
            y = target.to(device)
            
            pred = model(data)[:, 0]
            test_loss += np.sum((pred.numpy() - y.numpy())**2)
            size += len(y)
        test_rmse = np.sqrt(test_loss/size)
        print(f'test rmse: {test_rmse}')
        return test_rmse

def run(train_df, batch_size):
    for fold in range(n_splits):
        print(f'This is fold {fold} ....................')
        
        trigger_times = 0
        last_ts_loss = math.inf
        train_dataloader, val_dataloader = create_dataloader(train_df, batch_size, fold=fold)
        model, loss_fn, optimizer, device, scheduler = create_model()
        loss_df = pd.DataFrame()
        loss_df['epoch'] = list(range(n_epochs))
        loss_df['train_loss'] = [np.nan]*n_epochs
        loss_df['test_loss'] = [np.nan]*n_epochs
        for _epoch in range(n_epochs):
            print(f'This is epoch {_epoch} ...')
            tr_loss, model = train_model(train_dataloader, model, loss_fn, device, optimizer, scheduler)
            ts_loss = evaluate_model(val_dataloader, model, loss_fn, device)
            loss_df.loc[loss_df['epoch'] == _epoch, 'train_loss'] = tr_loss
            loss_df.loc[loss_df['epoch'] == _epoch, 'test_loss'] = ts_loss
            if ts_loss > last_ts_loss:
                trigger_times += 1
            if trigger_times > patience:
                print ("Early Stopping!...")
                break
            last_ts_loss = ts_loss
                
        torch.save(model, f'model_{fold}.pth')
        loss_df.to_csv(f'model_loss_{fold}.csv', index=False)

if __name__ == '__main__':
    train_df = load_CV_data()
    run(train_df, batch_size=batch_size)