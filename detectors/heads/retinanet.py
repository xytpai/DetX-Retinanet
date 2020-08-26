import torch
import torch.nn as nn 
import torch.nn.functional as F 
import math


class RetinaNetHead(nn.Module):
    def __init__(self, channels, num_classes, num_anchors=9):
        super(RetinaNetHead, self).__init__()
        self.conv_cls = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors*num_classes, kernel_size=3, padding=1))
        self.conv_reg = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_anchors*4, kernel_size=3, padding=1))
        for block in [self.conv_cls, self.conv_reg]:
            for layer in block.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.constant_(layer.bias, 0)
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
        pi = 0.01
        _bias = -math.log((1.0-pi)/pi)
        nn.init.constant_(self.conv_cls[-1].bias, _bias)
    
    def forward(self, x):
        return self.conv_cls(x), self.conv_reg(x)
