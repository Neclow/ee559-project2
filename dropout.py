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
        
    def __str__(self):
        '''
        Print main info on dropout layer, i.e. the probability of dropping out a node
        '''
        return f'{self.__class__.__name__}({self.p})'
        
    def forward(self, input_):
        if self.eval:
            # Dropout disabled during evaluation/testing mode
            return input_
        
        self.mask = (torch.rand(input_.shape) < self.p)/self.p
        return input_.mul(self.mask)

    def backward(self, G):
        return G.mul(self.mask)
