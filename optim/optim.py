import torch


class Optimizer(object):
    '''
    Optimization super-class

    Optimizers update parameter gradients after each forward+backward pass
    through gradient descent functions
    '''
    def step(self, *args):
        '''
        Gradient-based parameter update
        '''
        raise NotImplementedError


class SGD(Optimizer):
    '''
    Stochastic Gradient Descent optimization

    Attributes
    -------
    net
        Trained neural network
    eta
        Learning rate
    weight_decay
        L2-regularization coefficient
    momentum
        Momentum coefficient
    u
        Momentum vector
    '''
    def __init__(self, net, eta, weight_decay=0, momentum=0):
        self.net = net
        self.eta = eta
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.u = self.initialize_momenta() # Momentum "u"

    def initialize_momenta(self):
        '''
        Initialize momentum vector (as zeros)
        '''
        u = []
        for p in self.net.param():
            u.append(torch.zeros_like(p[0]))
        return u

    def step(self):
        '''
        Gradient-based parameter update

        If momentum is 0, then we fall back to vanilla SGD
        '''
        for i, p in enumerate(self.net.param()):
            self.u[i] = self.momentum*self.u[i] + self.eta*p[1]
            p[0].sub_(self.u[i])

            
class Adam(Optimizer):
    '''
    ADAptive Moment estimation (Adam) optimization (Kingma et al., 2014)

    Attributes
    -------
    net
        Trained neural network
    eta
        Learning rate
    beta
        Hyperparameters controlling decay rates (default values recommended by Kingma et al., 2014)
    eps
        Parameter to avoid divide-by-zero errors
    m
        Moving average of the gradient
    v
        Moving average of the squared gradient
    '''

    def __init__(self, net, eta, beta=(.9, .999), eps=1e-8):
        self.net = net
        self.eta = eta
        self.beta = beta
        self.eps = eps
        self.m = self.initialize()
        self.v = self.initialize()

    def initialize(self):
        '''
        Initialize moving average vectors (as zeros)
        '''
        x = []
        for p in self.net.param():
            x.append(torch.zeros_like(p[0]))
        return x

    def step(self):
        '''
        Gradient-based parameter update, following Adam algorithm
        '''
        for i, p in enumerate(self.net.param()):
            self.m[i] = self.beta[0]*self.m[i] + (1-self.beta[0])*p[1]
            self.v[i] = self.beta[1]*self.v[i] + (1-self.beta[1])*p[1].mul(p[1])

            m_hat = self.m[i] / (1-self.beta[0])
            v_hat = self.v[i] / (1-self.beta[1])
            
            p[0].sub_(self.eta * m_hat / (v_hat.sqrt() + self.eps))
