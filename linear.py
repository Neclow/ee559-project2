import torch
import math
from module import Module


class Linear(Module):
    '''
    A class for fully-connected layers.

    Attributes
    -------
    in_features
        Number of input features
    out_features
        Number of output features
    weight
        Layer weights, initalized by Gaussian He initialization
    bias
        Layer biases, initialized as zero
    dl_dw
        Gradient wrt the layer weights
    dl_db
        Gradient wrt the layer biases
    input_
        The input of the layer
    '''

    def __init__(self, in_features, out_features):
        '''
        Initialize the fully-connected layer

        Parameters
        -------
        in_features
            Number of input features
        out_features
            Number of output features
        '''

        super(Linear).__init__()
        std = math.sqrt(2/in_features)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.empty(out_features, in_features).normal_(0, std)
        self.bias = torch.zeros(out_features)

        self.dl_dw = torch.zeros(out_features, in_features)
        self.dl_db = torch.zeros(out_features)

    def __str__(self):
        '''
        Print main info on layer: number of input and output features
        '''
        return f'{self.__class__.__name__}({self.in_features}, {self.out_features})'

    def forward(self, input_):
        '''
        Run forward pass on input
        '''

        self.input_ = input_
        return self.input_.mm(self.weight.t()) + self.bias
    
    def weight_initialization(self, mode='kaiming'):
        if mode == 'kaiming':
            std = math.sqrt(2/self.in_features)
        elif mode == 'default':
            std = math.sqrt(1/self.in_features)
        else:
            raise ValueError('Weight initialization mode unknown. Try "kaiming" or "default".')
        
        self.weight = torch.empty(self.out_features, self.in_features).normal_(0, std)        

    def backward(self, G):
        '''
        Run backward pass (Back-propagation)
        Calculates gradients for both weights and biases of the layer.

        Parameters
        -------
        G
            Backpropagated gradient before the current layer

        Returns
        -------
        tensor
            Updated gradient through the derivative of the current layer
        '''
        self.dl_dw = G.t().mm(self.input_)
        self.dl_db = G.t().mv(torch.ones(G.shape[0]))
        return G.mm(self.weight)

    def zero_grad(self):
        '''
        Nullify gradients
        '''
        self.dl_dw = torch.zeros_like(self.dl_dw)
        self.dl_db = torch.zeros_like(self.dl_db)

    def param(self):
        '''
        Print parameters of layer

        Returns
        -------
        tensor
            List of pairs, each pair containing the parameter and its respective gradient
        '''

        return [(self.weight, self.dl_dw), (self.bias, self.dl_db)]
