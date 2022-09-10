
import joblib
import bayes_opt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

from data_pre import generate_data
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(8)
import optuna


import argparse
import glob
import json
import os
import platform
from datetime import datetime
from zipfile import ZipFile
# from create_dataset import ML1M_data

# from utils import DotDict, is_notebook

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
# from sharedbottom import SharedBottom
from deepctr_torch.models_mdl import SharedBottom
from deepctr_torch.models_mdl import MMoE





# put data into correct form
df = pd.read_csv('merged_m1.csv')
m1 = generate_data(df)
data, feature_names, dnn_feature_columns, sparse_features = m1.process()
# print(np.unique(data['gender'][:100]))
f = data[data['gender'] == 0]
train, test = train_test_split(data, test_size=0.1)

# train_0, test_0 = train_test_split(data[data['gender'] == 0][:100], test_size=0.2)
# train_1, test_1 = train_test_split(data[data['gender'] == 1][:100], test_size=0.2)
# train = train_0.append(train_1)
# test = test_0.append(test_1)

target = ['CT']
domain = ['gender']

output = {}
output['train'] = train
output['test'] = test
output['linear_feature_columns'] = dnn_feature_columns
output['dnn_feature_columns'] = dnn_feature_columns
# converts for DeepCTR to interpret
output['train_input'] = {name: train[name] for name in feature_names}
# converts for DeepCTR to interpret
output['test_input'] = {name: test[name] for name in feature_names}
output['train_output'] = train[target].values
output['test_output'] = test[target].values
output['train_domain'] = train[domain].values
output['test_domain'] = test[domain].values


# train_model_input = {name: train[name] for name in feature_names}
# test_model_input = {name: test[name] for name in feature_names}

# y_true = []
# y_0 = np.array(train[train['gender'] == 0]['CT'].values)
# y_1 = np.array(train[train['gender'] == 1]['CT'].values)
# z = np.full((len(y_1)-len(y_0), 1), 999)
# a = np.append(y_0, z)
# y_true = np.vstack((a, y_1)).T



# search_space = {
#     "lr": 0.005,
#     "batch_size": 256,
#     "cross_num": 4,
#     "l2_reg_embedding": 0.00005,
#     "l2_reg_linear": 0.00005,
#     "l2_reg_cross": 0.00005,
#     "l2_reg_dnn": 0.00005,
#     "dnn_hidden_units": 32,
#     "dnn_dropout": 0.5
# }
def objective(trial):
    search_space = {
        # "lr": tune.loguniform(1e-5, 1e-1),
        "lr": trial.suggest_loguniform("lr",1e-5, 1e-1),
        "batch_size": trial.suggest_categorical("batch_size", [16, 64, 256, 1024, 4098]),
        "cross_num": 4,
        "num_experts" : trial.suggest_categorical("dnn_hidden_units", [2,3,4,5]),
        "l2_reg_embedding": trial.suggest_uniform("l2_reg_embedding", 0.00000001, 0.0005),
        "l2_reg_linear": trial.suggest_uniform("l2_reg_linear", 0.00000001, 0.0005),
        "l2_reg_cross": 0.00005,
        "l2_reg_dnn": trial.suggest_uniform("l2_reg_dnn", 0.00000001, 0.0005),
        "dnn_hidden_units": 32,
        "dnn_dropout": trial.suggest_uniform("dnn_dropout",0.05,0.8 )
    }
    device = 'cpu'
    use_cuda = False
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    model = MMoE(output['linear_feature_columns'], output['dnn_feature_columns'],
                 task='binary', device=device, optimizer_hyper_parameters=search_space)

    model.compile("adam", "binary_crossentropy", metrics=['auc', 'logloss'], )
    history = model.fit(output['train_input'], output['train_output'],
                        batch_size=search_space['batch_size'], epochs=5, verbose=2, validation_split=0.2)
    auc = history.history['auc'][-1]
    return auc


study_name = 'Experiment_on_MMoE'

# Store and load using joblib:
study = optuna.create_study(study_name=study_name,direction="maximize")
joblib.dump(study, 'experiments.pkl')
study = joblib.load('experiments.pkl')
study.optimize(objective, n_trials=10)

##Predict on Test
best_params = study.best_params
best_model = MMoE(output['linear_feature_columns'], output['dnn_feature_columns'],
                  task='binary', device=device, optimizer_hyper_parameters=best_params)

best_model.compile("adam", "binary_crossentropy", metrics=['auc', 'logloss'], )
best_history = best_model.fit(output['train_input'], output['train_output'],
                    batch_size=best_model['batch_size'], epochs=30, verbose=2, validation_split=0.1)

pred = best_model.predict(output['test_input'], 256)


print("")
print("test LogLoss", round(log_loss(output['test_output'], pred), 4))
print("test AUC", round(roc_auc_score(output['test_output'], pred), 4))
