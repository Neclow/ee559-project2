from .optim import Optimizer
import torch

class Optimizer(object):
    def step(self, *args):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, net, eta, weight_decay=0, momentum=0, nesterov=False):
        self.net = net
        self.eta = eta
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.u = self.initialize_momenta() # Momentum "u"
        
       
    def initialize_momenta(self):
        u = []
        for i, p in enumerate(self.net.param()):
            u.append(torch.zeros_like(p[0]))
        return u
        
    def step(self):
        for i, p in enumerate(self.net.param()):            
            self.u[i] = self.momentum*self.u[i] + self.eta*p[1]
            p[0].sub_(self.u[i])
            #p[0].sub_(self.eta*p[1])