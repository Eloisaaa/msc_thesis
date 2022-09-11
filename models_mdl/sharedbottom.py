# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN


class SharedBottom(BaseModel):
    
    """Instantiates the Shared Bottom architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :domain_names: list containing integers, i.e. for three domains domain_names = [0,1,2]
    :optimizer_hyper_parameters: dictionary containing RayTune search space. 
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not DNN
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, optimizer_hyper_parameters, domain = None, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', 
                            dnn_use_bn=False,task='binary', device='cpu', gpus=None, domain_types=('binary', 'binary'), domain_names=[0,1]):

        super(SharedBottom, self).__init__(linear_feature_columns=linear_feature_columns,
                                  dnn_feature_columns=dnn_feature_columns, optimizer_hyper_parameters=optimizer_hyper_parameters,
                                  l2_reg_embedding=optimizer_hyper_parameters["l2_reg_embedding"], init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.dnn_hidden_units = (optimizer_hyper_parameters['dnn_hidden_units'], optimizer_hyper_parameters['dnn_hidden_units'])
        l2_reg_linear = optimizer_hyper_parameters['l2_reg_linear']
        l2_reg_cross = optimizer_hyper_parameters['l2_reg_cross']
        l2_reg_dnn = optimizer_hyper_parameters['l2_reg_dnn']

        # dnn_linear_in_feature = self.compute_input_dim(tower_output) # I think this should be dimension of tower output?
        dnn_linear_in_feature = self.dnn_hidden_units[-1] # for now......
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(
            device)

        self.shared_dnn = DNN(self.compute_input_dim(dnn_feature_columns), self.dnn_hidden_units,
                       activation=dnn_activation, use_bn=dnn_use_bn, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                       init_std=init_std, device=device)

        self.tower_dnn = DNN(dnn_linear_in_feature, self.dnn_hidden_units,
                       activation=dnn_activation, use_bn=dnn_use_bn, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                       init_std=init_std, device=device)

        # print(f'self.compute_input_dim(dnn_feature_columns): {self.compute_input_dim(dnn_feature_columns)}')

        ####### think about adding regularizer
        # self.add_regularization_weight(
        #     filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        # self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_linear)

        self.to(device)

        self.domain_types = domain_types
        self.domain_names = torch.tensor(domain_names)

        num_domains = len(domain_names)
        if num_domains <= 1:
            raise ValueError("num_domains must be greater than 1")
        if len(self.domain_types) != num_domains:
            raise ValueError("num_domains must be equal to the length of domain_types")

        for domain_type in self.domain_types:
            if domain_type not in ['binary', 'regression']:
                raise ValueError("domain must be binary or regression, {} is illegal".format(domain_type))

        self.tower_output = [[] for x in range(len(self.domain_names))]
        self.logit = [[] for x in range(len(self.domain_names))]
              
        
    def forward(self, X, domain_indicator, training):


        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        shared_bottom_output = self.shared_dnn(dnn_input)

        # this forward propagates through all towers based on the domain the data belongs to. 
        if training == True:
            y_pred = torch.empty(len(X),1)
            for domain in self.domain_names:
                domain_mask = torch.eq(domain,domain_indicator)
                # the following two lines seem a bit of a tedious way to mask by rows?.....
                masked_shared_bottom_output = domain_mask * torch.transpose(shared_bottom_output, 0, 1)
                masked_shared_bottom_output = torch.transpose(masked_shared_bottom_output, 0, 1)

                self.tower_output[domain] = self.tower_dnn(masked_shared_bottom_output)
                self.logit[domain] = self.dnn_linear(self.tower_output[domain])
                y_pred[domain_mask] = self.out(self.logit[domain])[domain_mask]
        # this forward propagates through a single domain given the unique domain the data belongs to (used for evaluation)
        elif training == False:
            domain = int(domain_indicator[0]) #  this will be the domain of the current batch
            self.tower_output[domain] = self.tower_dnn(shared_bottom_output)          
            self.logit[domain] = self.dnn_linear(self.tower_output[domain])
            y_pred = self.out(self.logit[domain])
        else:
            raise ValueError("training must be set to either True of False. ")

        return y_pred
