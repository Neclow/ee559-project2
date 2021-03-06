class Module(object):
    '''
    Default module class
    Modules can take an input and pass it through a forward pass,
    And update the back-propagated gradient
    Modules can also have parameters (i.e. weights and biases) that can be updated during training
    '''
    def __call__(self, *input_):
        '''
        Implement forward pass as the default method of a module
        '''
        return self.forward(*input_)
    
    def forward (self , *input_):
        '''
        Forward pass
        '''
        raise NotImplementedError
        
    def backward (self , *gradwrtoutput):
        '''
        Backward pass
        '''
        raise NotImplementedError
        
    def param (self):
        '''
        Returns list of parameters paired with their respective gradients
        '''
        return []