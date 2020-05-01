import torch
from module import Module

class tanh(Module):
    def __init__(self):
        super(tanh).__init__()
    
    def __str__(self):
        return f'{self.__class__.__name__}()'
        
    def forward(self, input_):
        self.input_ = input_
        return input_.tanh()
    
    def backward(self, G):
        return G.mul((1 - self.input_.tanh().pow(2)))
                     

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
