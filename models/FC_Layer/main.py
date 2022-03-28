import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim



class FC_Layer(nn.Module):
    def __init__(self, drop_out, num_classes, bias):
        super(FC_Layer, self).__init__()
        self.fc = nn.Linear(20, num_classes, bias = bias)
        self.layer = nn.Sequential(
            self.fc,
            nn.ReLU(),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        x = self.layer(x)

        return x