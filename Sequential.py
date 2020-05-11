from module import Module
from linear import Linear


class Sequential(Module):
    def __init__(self, modules):
        super(Sequential).__init__()
        
        self.modules = modules
    
    def __call__(self, input_):
        return self.forward(input_)
    
    def __str__(self):
        out = f'{self.__class__.__name__}(\n'
        for i, module in enumerate(self.modules):
            out += f'({i}) {module.__str__()},\n'
        out += f')\n'
        return out
    
    def param(self):
        param = []
        for module in self.modules:
            param.extend(module.param())
        return param
    
    def zero_grad(self):
        for module in self.modules:
            if isinstance(module, Linear):
                module.zero_grad()
    
    def forward(self, input_):
        for module in self.modules:
            input_ = module.forward(input_)
        return input_
    
    def backward(self, G):
        for module in reversed(self.modules):
            G = module.backward(G)