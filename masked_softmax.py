import math
from torch import nn
from torch.autograd import Function
import torch

class MaskedSoftmax(Function):
    @staticmethod
    def forward(ctx, prob, target, mask):
        N, C = prob.shape
        prob[~mask] = -1e45
        p = np.exp(prob - np.max(prob, -1, keepdims=True))
        p = prob / torch.sum(p * mask, dim=-1, keepdim=True)
        ctx.save_for_backward(p)

        one_hot = np.zeros((N, C), dtype=bool)
        one_hot[range(len(y)), y] = 1

        return - np.log(p[one_hot]).mean()

    @staticmethod
    def backward(ctx, grad_output):
        return dp_cpp.backward(grad_output, *ctx.saved_variables), None, None, None

masked_softmax = MaskedSoftmax.apply
