import torch
from module import Module

class MSELoss(Module):
    '''
    Mean square error loss function

    Attributes
    -------
    input_
        The input of the loss function (predicted target)
    target
        The real corresponding target/label to the input
    '''

    def __init__(self):
        super(MSELoss).__init__()

    def forward(self, input_, target):
        '''
        Run forward pass

        Parameters
        -------
        input_
            Input of the layer (i.e. predicted target)
        target
            Target/label corresponding to the input

        Returns
        -------
        tensor
            Loss/cost of the prediction
        '''

        self.input_ = input_
        self.target = target
        return (input_ - target).pow(2).mean()

    def backward(self):
        '''
        Run the backward pass (Back-propagation), i.e. the derivative of the loss function
        '''
        return 2*(self.input_ - self.target)/(self.input_.shape[0])

class CrossEntropyLoss(Module):
    '''
    Cross-entropy loss function
    This function includes softmax activation of the input,
    as it facilitates gradient calculations

    Attributes
    -------
    softmax_input
        The input of the loss function (predicted target), activated by softmax
    target
        The real corresponding target/label to the input
    '''
    def __init__(self):
        super(CrossEntropyLoss).__init__()

    def forward(self, input_, target):
        '''
        Run forward pass

        Parameters
        -------
        input_
            Input of the layer (i.e. predicted target)
        target
            Target/label corresponding to the input

        Returns
        -------
        tensor
            Loss/cost of the prediction
        '''
        self.softmax_input = input_.exp()/(input_.exp().sum(1).repeat(2,1).t())
        self.target = target

        return -self.target.mul(self.softmax_input.log()).sum(1).mean()

    def backward(self):
        '''
        Run the backward pass (Back-propagation), i.e. the derivative of the loss function
        '''
        return -(self.target-self.softmax_input)
