# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:16:40 2022

@author: jiyoon
"""

import pandas as pd
import numpy as np
import main_classificaiton as mc
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import random

#%%
random_seed = 42

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

#%%

# Case 1. w/o data representation & RNN model 
config1 = {
        'with_representation': False, # classification에 사용되는 representation이 있을 경우 True, 아닐 경우 False
        'model': 'RNN', # classification에에 활용할 알고리즘 정의, {'RNN', 'LSTM', 'GRU', 'CNN_1D', 'FC_layer'} 중 택 1

        'parameter': {
            'window_size' : 50, # input time series data를 windowing 하여 자르는 길이(size)
            'num_layers' : 2, # recurrnet layers의 수, Default : 1
            'hidden_size' : 64, # hidden state의 벡터차원 수
            'attention' : False, # True일 경우 attention layer를 추가
            'dropout' : 0.2, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'bidirectional' : True, # 모델의 양방향성 여부
            'batch_size' : 64, #batch size
            'device': 'cuda', # 학습 환경, ["cuda", "cpu"] 중 선택
            'num_epochs' : 200 # 학습 시 사용할 epoch 수
            }
}
#%%
data_dir = './data'

train_x = pickle.load(open(os.path.join(data_dir, 'x_train.pkl'), 'rb'))
train_y = pickle.load(open(os.path.join(data_dir, 'state_train.pkl'), 'rb'))
test_x =  pickle.load(open(os.path.join(data_dir, 'x_test.pkl'), 'rb'))
test_y = pickle.load(open(os.path.join(data_dir, 'state_test.pkl'), 'rb'))

train_data = {'x' : train_x, 'y' : train_y}
test_data = {'x' : test_x, 'y' : test_y}

#%%

# Case 1. w/o data representation & RNN
config = config1
data_classification = mc.Classification(config, train_data, test_data)
test_accuracy = data_classification.getResult()
