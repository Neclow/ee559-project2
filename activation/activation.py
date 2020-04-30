import torch
from module import Module

class tanh(Module):
    def __init__(self):
        super(tanh).__init__()
    
    def __str__(self):
        return f'{self.__class__.__name__}()'
        
    def forward(self, x):
        return x.tanh()
    
    def backward(self, x):
        return 1 - x.tanh().pow(2)

class ReLU(Module):
    def __init__(self):
        super(ReLU).__init__()
    
    def __str__(self):
        return f'{self.__class__.__name__}()'
    
    def __call__(self, input_):
        return self.forward(input_)
    
    def forward(self, input_):
        self.input_ = input_
        return input_.relu()

    def backward(self, G):
        return G.mul((self.input_ > 0).float())
       
    def param(self):
        return []
