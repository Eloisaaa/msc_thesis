import pandas as pd
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras_preprocessing.sequence import pad_sequences
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DeepFM, DCN, xDeepFM

class generate_data():
    def __init__(self,data):
        self.data = data
        self.key2index = {}


    def data_process(self,data_df, dense_features, sparse_features):
        # Replace continuous NA data to 0.0
        data_df[dense_features] = data_df[dense_features].fillna(0.0)
        # Replace discrete NA data to -1
        data_df[sparse_features] = data_df[sparse_features].fillna("-1")
        for feat in sparse_features:
            lbe = LabelEncoder()
            data_df[feat] = lbe.fit_transform(data_df[feat])
        return data_df[dense_features+sparse_features]


    """ 1. Generate the paded and encoded sequence feature of sequence input feature(value 0 is for padding).
        2. Generate config of sequence feature with VarLenSparseFeat """


    def split(self,x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in self.key2index:
                # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
                self.key2index[key] = len(self.key2index) + 1
        return list(map(lambda x: self.key2index[x], key_ans))

    def process(self):
        columns = self.data.columns.values
        dense_features = ["timestamp"]
        sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
        target = ['CT']
        domain = ['gender']  # domain is defined by gender

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            self.data[feat] = lbe.fit_transform(self.data[feat])

        mms = MinMaxScaler(feature_range=(0, 1))  # 最大最小值标准化
        self.data[dense_features] = mms.fit_transform(self.data[dense_features])

        # Preprocess the sequence feture(padding)
        # list([1]), list([1, 4]), list([1, 4, 11]), list([1, 5]),list([1, 5, 10]), list([1, 5, 11])...
        genres_list = list(map(self.split, self.data['genres'].values))
        genres_length = np.array(list(map(len, genres_list)))  # 1,2,3,2,3,3 ....
        max_len = max(genres_length)  # 6 which means the max genres contains 6 genres
        # Notice : padding=`post`
        genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )
        '''
        genres_list = array([[ 1,  0,  0,  0,  0,  0],
            [ 1,  0,  0,  0,  0,  0],
            [ 1,  0,  0,  0,  0,  0],
            ...,
            [ 1,  0,  0,  0,  0,  0],
            [ 6,  1, 15,  0,  0,  0],
            [18,  0,  0,  0,  0,  0]],
        '''
        # 2.count #unique features for each sparse field and generate feature config for sequence feature

        fixlen_feature_columns = [SparseFeat(feat, self.data[feat].nunique(), embedding_dim=4)
                                for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                for feat in dense_features]

        varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
            self.key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean')]  # Notice : value 0 is for padding for sequence input feature

        linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        sparse_features.append('genre')

        self.data['genres'] = genres_list


        return self.data, feature_names, dnn_feature_columns, sparse_features
