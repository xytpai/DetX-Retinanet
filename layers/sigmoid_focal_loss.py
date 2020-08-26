import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import sigmoid_focal_loss_cuda


class _sigmoid_focal_loss(Function):
    @staticmethod
    def forward(ctx, input, target, gamma=2.0, alpha=0.25):
        ctx.save_for_backward(input, target)
        num_classes = input.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha
        return sigmoid_focal_loss_cuda.forward(input, target, 
                    num_classes, gamma, alpha)
        
    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        input, target = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_input = sigmoid_focal_loss_cuda.backward(input, target, d_loss,
                                                    num_classes, gamma, alpha)
        return d_input, None, None, None, None
    
sigmoid_focal_loss = _sigmoid_focal_loss.apply


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        loss = sigmoid_focal_loss(logits, targets, self.gamma, self.alpha)
        return loss.sum()
