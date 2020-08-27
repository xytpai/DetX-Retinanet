import torch
import torch.nn as nn 
import torch.nn.functional as F 
from layers import *
import math


def tlbr2yxyx_anchor_yxyx(reg, acr, factor=1.0):
    '''
    reg: F(..., 4)
    acr: F(..., 4) yxyx
    ->   F(..., 4)
    '''
    acr_hw = acr[..., 2:] - acr[..., :2] + 1
    ctr_yx = (acr[..., :2] + acr[..., 2:])/2.0
    ymin_xmin = ctr_yx + reg[..., :2]*acr_hw*factor
    ymax_xmax = ctr_yx + reg[..., 2:]*acr_hw*factor
    return torch.cat([ymin_xmin, ymax_xmax], dim=-1)


class RetinaNetHead(nn.Module):
    def __init__(self, channels, num_class, norm=1/8.0, num_anchors=9):
        super(RetinaNetHead, self).__init__()
        self.channels = channels
        self.num_class = num_class
        self.norm = norm
        self.num_anchors = num_anchors
        self.conv_cls = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_anchors*num_class, kernel_size=3, padding=1))
        self.conv_reg = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_anchors*4, kernel_size=3, padding=1))
        for block in [self.conv_cls, self.conv_reg]:
            for layer in block.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.constant_(layer.bias, 0)
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
        pi = 0.01
        _bias = -math.log((1.0-pi)/pi)
        nn.init.constant_(self.conv_cls[-1].bias, _bias)
    
    def forward(self, x, im_h, im_w, anchors):
        '''
        x: F(b, c, h_s, w_s)
        anchors: F(n, 2) # h, w

        Return: 
        cls_s: F(b, h_s*w_s*n, num_class)
        reg_s: F(b, h_s*w_s*n, 4) ymin, xmin, ymax, xmax
        acr_s: F(b, h_s*w_s*n, 4) ymin, xmin, ymax, xmax
        '''
        batch_size, c, h_s, w_s = x.shape
        stride = (im_h-1) // (h_s-1)
        acr_s = anchor_scatter(anchors*stride*self.norm, batch_size, h_s, w_s, stride) 
        cls_s = self.conv_cls(x)
        reg_s = self.conv_reg(x)
        cls_s = cls_s.permute(0,2,3,1).contiguous()
        reg_s = reg_s.permute(0,2,3,1).contiguous()
        cls_s = cls_s.view(batch_size, -1, self.num_class)
        reg_s = reg_s.view(batch_size, -1, 4)
        reg_s = tlbr2yxyx_anchor_yxyx(reg_s, acr_s)
        return cls_s, reg_s, acr_s
