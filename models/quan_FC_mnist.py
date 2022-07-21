import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .quantization import *
__all__=['FC','FC_3_quan']

class FC(nn.Module):

    def __init__(self, n_classes):
        super(FC, self).__init__()

        self.features = nn.Sequential(
            quan_Linear(in_features=1024,out_features=500),
            nn.ReLU(),
            quan_Linear(in_features=500, out_features=150),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            quan_Linear(in_features=150, out_features=n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
def FC_3_quan(pretrained=False, **kwargs):
    model = FC(**kwargs,n_classes=10)
    if pretrained:
        model = torch.load("/media/hamid/ali/RESEARCH/BFA-master/save/quan_fc_mnist/fc.pth")

    return model

