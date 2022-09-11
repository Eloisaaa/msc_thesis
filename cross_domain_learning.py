from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from data_pre import generate_data
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(8)

import argparse
import glob
import json
import os
import platform
from datetime import datetime
from zipfile import ZipFile

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
# from sharedbottom import SharedBottom
from deepctr_torch.models_mdl import SharedBottom
from deepctr_torch.models_mdl import MMoE





# put data into correct form
df = pd.read_csv('/Users/apple/Desktop/cross_domain/merged_m1.csv')
m1 = generate_data(df)
data, feature_names, dnn_feature_columns, sparse_features = m1.process()
f = data[data['gender'] == 0]
train, test = train_test_split(data, test_size=0.2)

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



device = 'cpu'
use_cuda = False
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

search_space = {
    "lr": 0.005,
    "batch_size": 256,
    "cross_num": 4,
    "l2_reg_embedding": 0.00005,
    "l2_reg_linear": 0.00005,
    "l2_reg_cross": 0.00005,
    "l2_reg_dnn": 0.00005,
    "dnn_hidden_units": 32,
    "dnn_dropout": 0.5
}
model = SharedBottom(dnn_feature_columns, dnn_feature_columns, task='binary',
                    device=device, optimizer_hyper_parameters = search_space)
model = MMoE(output['linear_feature_columns'], output['dnn_feature_columns'], num_experts = 3, task='binary',
                    device=device, optimizer_hyper_parameters = search_space)

model.compile("adam", "binary_crossentropy", metrics=['auc', 'logloss'], )
history = model.fit(output['train_input'], output['train_output'],
                    batch_size=256, epochs=1, verbose=2, validation_split=0.2)

pred = model.predict(output['test_input'], 256)


print("")
print("test LogLoss", round(log_loss(output['test_output'], pred), 4))
print("test AUC", round(roc_auc_score(output['test_output'], pred), 4))
