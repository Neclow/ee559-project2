import torch
from module import Module

class MSELoss(Module):
    def __init__(self):
        super(MSELoss).__init__()

    def __call__(self, input_, target):
        return self.forward(input_, target)
    
    def forward(self, input_, target):
        self.input_ = input_
        self.target = target
        return (input_ - target).pow(2).mean()
    
    def backward(self):
        return 2*(self.input_ - self.target)/(self.input_.shape[0])