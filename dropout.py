from module import Module
import torch


class InvertedDropout(Module):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, p):
        return self.forward(p)
    
    def forward(self, input_):
        self.mask = (torch.rand(input_.shape) < self.p)/self.p
        return input_.mul(self.mask)
    
    def backward(self, G):
        return G.mul(self.mask)