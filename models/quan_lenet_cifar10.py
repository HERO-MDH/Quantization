import numpy as np
import torch
import torch.nn as nn
from .quantization import *
__all__=['LeNet5_cifar','lenet_5_quan_cifar']
class LeNet5_cifar(nn.Module):

    def __init__(self, n_classes=10):
        super(LeNet5_cifar, self).__init__()
        self.features = nn.Sequential(
            quan_Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1),
            nn.Relu(),
            nn.MaxPool2d(kernel_size=2),
            quan_Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1),
            nn.Relu(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            quan_Linear(in_features=5*5*50, out_features=500),
            nn.Tanh(),
            quan_Linear(in_features=500, out_features=n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
def lenet_5_quan_cifar(pretrained=False, **kwargs):

    model = LeNet5_cifar(**kwargs)
    if pretrained:
        model = torch.load("/media/hamid/ali/RESEARCH/DeepFool_weight_attack(ICCAD)/model_robust.pth")

    return model
