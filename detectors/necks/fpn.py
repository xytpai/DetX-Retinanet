import torch
import torch.nn as nn 
import torch.nn.functional as F 
from layers import conv_with_kaiming_uniform


class FPN(nn.Module):
    def __init__(self, in_channels=[512,1024,2048], out_channel=256, 
                            top_mode='LastLevelP6P7'):
        super(FPN, self).__init__()
        if top_mode == 'LastLevelP6P7':
            self.top_blocks = LastLevelP6P7(out_channel, out_channel)
        elif top_mode == 'LastLevelMaxPool':
            self.top_blocks = LastLevelMaxPool()
        else:
            self.top_blocks = None
        Conv2d = conv_with_kaiming_uniform(use_gn=False, use_relu=False)
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels):
            inner_block = "fpn_inner{}".format(idx) # fpn_inner0, fpn_inner1, ...
            layer_block = "fpn_layer{}".format(idx)
            inner_block_module = Conv2d(in_channels, out_channel, kernel_size=1)
            layer_block_module = Conv2d(out_channel, out_channel, kernel_size=3)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        
    def forward(self, x):
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            inner_lateral = getattr(self, inner_block)(feature)
            inner_top_down = F.interpolate(
                last_inner, size=(int(inner_lateral.shape[-2]), int(inner_lateral.shape[-1])),
                mode='bilinear', align_corners=True)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))
        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)
        return results


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
