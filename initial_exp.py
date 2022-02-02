# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:13:17 2022

@author: snourashrafeddin
"""


import pandas as pd
import numpy as np 
import optuna
import lightgbm
from lightgbm import LGBMRegressor
from optuna.samplers import TPESampler

np.random.seed(0)
sampler = TPESampler(seed=0)

test_df = pd.read_csv('example_test.csv')
# submission_df = pd.read_csv('example_sample_submission.csv')
train_df = pd.read_parquet('train_low_mem.parquet')

n_features = 300
y_col = 'target'
X_cols = ['investment_id'] + [f'f_{i}' for i in range(n_features)]

def train_models(train_df, k=3, n_trails=50):
    studies = []
    
    def LGBM_objective(trial):
        param = {
            'boosting_type':trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1024),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-5, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-5, 10.0),
            'learning_rate': trial.suggest_float('learning_rate', 1e-10, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 8, 256),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
            'min_child_samples': 10,
            'boost_from_average' : False
        }
       
        model = LGBMRegressor(**param)
        #sample_weight = np.arange(len(train_data))+1
        #sample_weight = list(sample_weight/np.sum(sample_weight))
        model.fit(train_df[random_selection][X_cols], train_df[random_selection][y_col], 
                  eval_set=[(train_df[~random_selection][X_cols], train_df[~random_selection][y_col])], early_stopping_rounds=50)
        pred = model.predict(train_df[~random_selection][X_cols])
        #score = np.sqrt(np.mean(train_df[~random_selection][y_col] - pred) **2)
        score = np.mean(np.abs(train_df[~random_selection][y_col] - pred))
        return score
    for i in range(k):
        random_selection = np.random.rand(len(train_df.index)) <= 0.80
        #train_data = train_df[random_selection]
        #valid_data = train_df[~random_selection]

        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(LGBM_objective, n_trials=n_trails)
        studies.append(study)
        
    #sample_weight = np.arange(len(train_df))+1
    #sample_weight = list(sample_weight/np.sum(sample_weight))
    vals = []
    models = []
    for study in studies:
        model = LGBMRegressor(**study.best_trial.params)
        model.fit(train_df[X_cols], train_df[y_col])
        models.append(model) 
        vals.append(study.best_trial.value)

    model_weights = 1/np.array(vals)
    model_weights = model_weights/np.sum(model_weights)
    model_weights
    
    return models, model_weights, vals
    
models, model_weights, vals = train_models(train_df, k=3)

for i in range(len(models)):
    models[i].booster_.save_model(f'./models/model{i}.txt')

with open('./models/weights.npy', 'wb') as f:
    np.save(f, model_weights)
    
with open('./models/vals.npy', 'wb') as f:
    np.save(f, vals)


with open('./models/weights.npy', 'rb') as f:
    a = np.load(f)
    
save_models = []
for i in range(len(models)):
    save_models.append(lightgbm.Booster(model_file=f'./models/model{i}.txt'))

# pred = np.array([0]*len(test_df))
# for i in range(len(models)):
#     pred = pred + model_weights[i]*models[i].predict(test_df[X_cols])

pred = np.array([0]*len(test_df))
for i in range(len(models)):
    pred = pred + model_weights[i]*models[i].predict(test_df[X_cols])