import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim



class CNN_1D(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, drop_out, num_classes):
        super(CNN_1D, self).__init__()
        # 첫 번째 1-dimensional convolution layer 구축
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )
        # 두 번째 1-dimensional convolution layer 구축
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, output_channels, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )
        # fully-connected layer 구축
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(output_channels * 11, num_classes) # 이부분은 hyperparameter에 따라 계산을 해줘야 함

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x