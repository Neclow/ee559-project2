from module import Module
import torch


class InvertedDropout(Module):
    '''
    Inverted Dropout regularization
    (Parameter-less module)

    Contrary to Dropout, inverted Dropout scales activations at train time

    Attributes
    -------
    p
        Amount of neurons to be dropped
    eval
        If true, disable layer (necessary during test time)
    '''
    def __init__(self, p=0.5):
        self.p = p
        self.eval = False
        
    def forward(self, input_):
        if eval:
            return input_
        self.mask = (torch.rand(input_.shape) < self.p)/self.p
        return input_.mul(self.mask)

    def backward(self, G):
        return G.mul(self.mask)
