import torch
from module import Module


class tanh(Module):
    '''
    Hyperbolic tangent activation
    N.B. Activation functions are parameter-less
    => Thus, activation functions inherit "param" function from superclass *module*

    Attributes
    -------
    input_
        The input before activation
    '''
    def __init__(self):
        super(tanh).__init__()

    def __str__(self):
        '''
        Print the class name
        '''
        return f'{self.__class__.__name__}()'

    def forward(self, input_):
        '''
        Run forward pass

        Parameters
        -------
        input_
            Input of the layer

        Returns
        -------
        tensor
            Activated input
        '''
        self.input_ = input_
        return input_.tanh()

    def backward(self, G):
        '''
        Run backward pass (Back-propagation)

        Parameters
        -------
        G
            Backpropagated gradient before activation

        Returns
        -------
        tensor
            Backpropagated gradient after activation
        '''
        return G.mul((1 - self.input_.tanh().pow(2)))

    
class ReLU(Module):
    '''
    Rectified Linear Unit (ReLU) activation
    N.B. Activation functions are parameter-less
    => Thus, activation functions inherit "param" function from superclass *module*

    Attributes
    -------
    input_
        The input before activation
    '''
    def __init__(self):
        super(ReLU).__init__()

    def __str__(self):
        '''
        Print the class name
        '''
        return f'{self.__class__.__name__}()'

    def forward(self, input_):
        '''
        Run forward pass

        Parameters
        -------
        input_
            Input of the layer

        Returns
        -------
        tensor
            Activated input
        '''
        self.input_ = input_
        return input_.clamp(min=0)

    def backward(self, G):
        '''
        Run backward pass (Back-propagation)

        Parameters
        -------
        G
            Backpropagated gradient before activation

        Returns
        -------
        tensor
            Backpropagated gradient after activation
        '''
        return G.mul((self.input_ > 0).float())
