import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn



def create_classification_dataset(window_size, train_x, train_y, test_x, test_y, batch_size):
    # data_dir에 있는 train/test 데이터 불러오기
    # x = pickle.load(open(os.path.join(data_dir, 'x_train.pkl'), 'rb'))
    # y = pickle.load(open(os.path.join(data_dir, 'state_train.pkl'), 'rb'))
    # x_test = pickle.load(open(os.path.join(data_dir, 'x_test.pkl'), 'rb'))
    # y_test = pickle.load(open(os.path.join(data_dir, 'state_test.pkl'), 'rb'))

    x = train_x
    y = train_y
    x_test = test_x
    y_test = test_y
    
    # train data를 시간순으로 8:2의 비율로 train/validation set으로 분할
    n_train = int(0.8 * len(x))
    n_valid = len(x) - n_train
    n_test = len(x_test)
    x_train, y_train = x[:n_train], y[:n_train]
    x_valid, y_valid = x[n_train:], y[n_train:]

    # train/validation/test 데이터를 window_size 시점 길이로 분할
    datasets = []
    for set in [(x_train, y_train, n_train), (x_valid, y_valid, n_valid), (x_test, y_test, n_test)]:
        T = set[0].shape[-1]
        windows = np.split(set[0][:, :, :window_size * (T // window_size)], (T // window_size), -1)
        windows = np.concatenate(windows, 0)
        labels = np.split(set[1][:, :window_size * (T // window_size)], (T // window_size), -1)
        labels = np.round(np.mean(np.concatenate(labels, 0), -1))
        datasets.append(torch.utils.data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))

    # train/validation/test DataLoader 구축
    trainset, validset, testset = datasets[0], datasets[1], datasets[2]
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader