from module import Module
from linear import Linear


class Sequential(Module):
    '''
    A class to build sequences of modules

    Attributes
    -------
    modules: a list of modules
    '''
    def __init__(self, modules):
        super(Sequential).__init__()
        self.modules = modules

    def __str__(self):
        '''
        Fancy print of each module in a Sequential
        '''
        out = f'{self.__class__.__name__}(\n'
        for i, module in enumerate(self.modules):
            out += f'({i}) {module.__str__()},\n'
        out += f')\n'
        return out

    def param(self):
        '''
        Print parameters of layer

        Returns
        -------
        param
            List of pairs, each pair containing the parameter and its respective gradient
        '''
        param = []
        for module in self.modules:
            param.extend(module.param())
        return param

    def zero_grad(self):
        '''
        Nullify gradients
        '''
        for module in self.modules:
            if isinstance(module, Linear):
                module.zero_grad()

    def train(self):
        '''
        Activate training mode, i.e, enable Dropout layers (if there are any)
        '''
        for module in self.modules:
            if isinstance(module, InvertedDropout):
                module.eval = False

    def eval(self):
        '''
        Activate testing mode, i.e, disable Dropout layers (if there are any)
        '''
        for module in self.modules:
            if isinstance(module, InvertedDropout):
                module.eval = True

    def forward(self, input_):
        '''
        Run forward pass on input
        '''
        for module in self.modules:
            input_ = module.forward(input_)
        return input_

    def backward(self, G):
        '''
        Run backward pass (Back-propagation)
        Calculates gradients for both weights and biases of the layer.

        Parameters
        -------
        G
            Backpropagated gradient before the current layer
            (Here, the gradient of losses wrt forward pass output)
        '''
        for module in reversed(self.modules):
            G = module.backward(G)
