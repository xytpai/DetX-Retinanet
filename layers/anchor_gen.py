import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Function
import anchor_gen_cuda


class _anchor_scatter(Function):
    @staticmethod
    def forward(ctx, anchors, batch_size, ph, pw, stride, to_move=0.0):
        '''
        anchors: F(an, 2) h,w
        ->       F(b, hwan, 4) yxyx
        '''
        assert anchors.dtype == torch.float
        assert anchors.is_cuda
        assert anchors.shape[1] == 2
        return anchor_gen_cuda.anchor_scatter(anchors, 
                    batch_size, ph, pw, stride, to_move)
    @staticmethod
    def symbolic(g, *inputs):
        return g.op("anchor_scatter", inputs[0])
anchor_scatter = _anchor_scatter.apply


class _center_scatter(Function):
    @staticmethod
    def forward(ctx, anchors, batch_size, ph, pw, stride, to_move=0.0):
        '''
        anchors: F(n) 
	    ->       F(b, hw, 2+n) # 2+n: cy, cx, ...
        '''
        assert anchors.dtype == torch.float
        assert anchors.is_cuda
        assert anchors.dim() == 1
        return anchor_gen_cuda.center_scatter(anchors, 
                    batch_size, ph, pw, stride, to_move)
    @staticmethod
    def symbolic(g, *inputs):
        return g.op("center_scatter", inputs[0])
center_scatter = _center_scatter.apply
