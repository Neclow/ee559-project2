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

class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss).__init__()
    
    def __call__(self, input_, target):
        return self.forward(input_, target)
    
    def forward(self, input_, target):
        self.softmax_input = input_.exp()/(input_.exp().sum(1).repeat(2,1).t())
        self.target = target
        
        return -self.target.mul(self.softmax_input.log()).sum(1).mean()
    
    def backward(self):
        return -(self.target-self.softmax_input)