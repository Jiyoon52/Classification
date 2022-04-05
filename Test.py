# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 23:23:45 2022

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

# seed 고정
random_seed = 42

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

#%%

# Case 4. w/o data representation & CNN_1D model 
config4 = {
        'with_representation': False, # classification에 사용되는 representation이 있을 경우 True, 아닐 경우 False
        'model': 'CNN_1D', # classification에에 활용할 알고리즘 정의, {'RNN', 'LSTM', 'GRU', 'CNN_1D', 'FC_layer'} 중 택 1

        'parameter': {
            'window_size' : 128, # input time series data를 windowing 하여 자르는 길이(size)
            'output_channels' : 64, # convolution channel size of output
            'drop_out' : 0.2, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
            'kernel_size' : 3, # convolutional filter size
            'stride' : 1, # stride of the convolution. Default = 1 
            'padding' : 0, # padding added to both sides of the input. Default = 0
            'batch_size' : 64, # batch size
            'device': 'cuda', # 학습 환경, ["cuda", "cpu"] 중 선택
            'num_epochs' : 150 # 학습 시 사용할 epoch 수
            }
}

#%%
data_dir = './data'

train_x = pickle.load(open(os.path.join(data_dir, 'X_train.pkl'), 'rb'))
train_y = pickle.load(open(os.path.join(data_dir, 'y_train.pkl'), 'rb'))
test_x =  pickle.load(open(os.path.join(data_dir, 'X_test.pkl'), 'rb'))
test_y = pickle.load(open(os.path.join(data_dir, 'y_test.pkl'), 'rb'))

train_data = {'x' : train_x, 'y' : train_y}
test_data = {'x' : test_x, 'y' : test_y}

print(train_x.shape)  #shape : (num_of_instance x input_dims x window_size) = (7352, 9, 128)
print(train_y.shape) #shape : (num_of_instance x input_dims x window_size) = (7352, )
print(test_x.shape)  #shape : (num_of_instance x input_dims x window_size) = (2947, 9, 128)
print(test_y.shape)  #shape : (num_of_instance x input_dims x window_size) = (2947)

#%%

# Case 4. w/o data representation & CNN_1D
config = config4
data_classification = mc.Classification(config, train_data, test_data)

train_loader, valid_loader, test_loader = data_classification.get_loaders( train_data, test_data, 128, 64, False)
txt, labels = next(iter(train_loader))

pred, prob = data_classification.getResult()
