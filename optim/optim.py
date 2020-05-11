import torch


class Optimizer(object):
    def step(self, *args):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, net, eta, weight_decay=0, momentum=0):
        self.net = net
        self.eta = eta
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.u = self.initialize_momenta() # Momentum "u"       
       
    def initialize_momenta(self):
        u = []
        for p in self.net.param():
            u.append(torch.zeros_like(p[0]))
        return u
        
    def step(self):
        for i, p in enumerate(self.net.param()):
            self.u[i] = self.momentum*self.u[i] + self.eta*p[1]
            p[0].sub_(self.u[i])

class Adam(Optimizer):
    def __init__(self, net, eta, beta=(.9, .999), eps=1e-8):
        self.net = net
        self.eta = eta
        self.beta = beta
        self.eps = eps
        self.m = self.initialize()
        self.v = self.initialize()
    
    def initialize(self):
        x = []
        for p in self.net.param():
            x.append(torch.zeros_like(p[0]))
        return x
   
    def step(self):
        for i, p in enumerate(self.net.param()):
            self.m[i] = self.beta[0]*self.m[i] + (1-self.beta[0])*p[1]
            self.v[i] = self.beta[1]*self.v[i] + (1-self.beta[1])*p[1].mul(p[1])
            
            m_hat = self.m[i] / (1-self.beta[0])
            v_hat = self.v[i] / (1-self.beta[1])
            
            p[0].sub_(self.eta * m_hat / (v_hat.sqrt() + self.eps))
            