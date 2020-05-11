import torch
import math
from module import Module


class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear).__init__()
        std = 1/math.sqrt(in_features)
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = torch.empty(out_features, in_features).normal_(0, std)
        self.bias = torch.zeros(out_features)
        
        self.dl_dw = torch.zeros(out_features, in_features)
        self.dl_db = torch.zeros(out_features)
       
    def __str__(self):
        return f'{self.__class__.__name__}({self.in_features}, {self.out_features})'
    
    def __call__(self, input_):
        return self.forward(input_)
        
    def forward(self, input_):
        self.input_ = input_
        return self.input_.mm(self.weight.t()) + self.bias
        
    def backward(self, G):
        self.dl_dw = G.t().mm(self.input_)
        self.dl_db = G.t().mv(torch.ones(G.shape[0]))
        
        return G.mm(self.weight)
    
    def zero_grad(self): 
        self.dl_dw = torch.zeros_like(self.dl_dw)
        self.dl_db = torch.zeros_like(self.dl_db)
    
    def param(self):
        return [(self.weight, self.dl_dw), (self.bias, self.dl_db)]