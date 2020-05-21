from module import Module
from linear import Linear
from dropout import InvertedDropout


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
    
    def __len__(self):
        '''
        Print number of modules in Sequential
        '''
        return len(self.modules)
    
    
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
    
    def weight_initialization(self):
        '''
        Initialize weights of fully-connected layers
        '''
        
        # Find list of module names in Sequential
        module_names = [module.__class__.__name__.lower() for module in self.modules]
        
        # Update corrective gain for relu/tanh activation
        # otherwise pick linear gain (1.0) for weight initialization()
        if 'tanh' in module_names:
            gain = 'tanh'
        elif 'relu' in module_names:
            gain = 'relu'
        else:
            gain = 'lin'
        
        for module in self.modules:
            if isinstance(module, Linear):
                module.weight_initialization(gain)

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
                
    
