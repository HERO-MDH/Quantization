import numpy as np
import torch
import torch.nn as nn
from .quantization import *
__all__=['LeNet5','lenet_5_quan']
class LeNet5(nn.Module):

    def __init__(self, n_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            quan_Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            quan_Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            quan_Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            quan_Linear(in_features=120, out_features=84),
            nn.Tanh(),
            quan_Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
def lenet_5_quan(pretrained=False, **kwargs):

    model = LeNet5(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load('pretrain_models/lenet_5_quan.pth'))

    return model