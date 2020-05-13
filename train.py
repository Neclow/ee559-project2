import torch
from loss import *
from optim import *


def train(model, trainX, trainY, input_criterion = 'mse', input_optimizer = 'sgd',
          nb_epochs = 250, eta=1e-3, mini_batch_size=100, verbose=False):
    '''
    Network training

    Parameters
    -------
    model
        Neural network to train
    trainX
        Training data examples
    trainY
        Training data labels
    input_criterion
        String to choose loss function
        'mse': MSE loss
        'cross': Cross-entropy loss
    input_optimizer
        String to choose optimizer
        'sgd': SGD
        'mom': SGD with momentum (0.9)
        'adam': Adam
    nb_epochs
        Number of training epochs
    Eta
        Learning rate
    Mini_batch_size
        Size of mini-batch during training
    verbose
        If true, prints loss every 10 epochs

    Returns
    -------
    Computed training loss at each epoch
    '''

    # Disable Pytorch autograd
    torch.set_grad_enabled(False)

    if input_criterion == 'mse':
        criterion = MSELoss()
    elif input_criterion == 'cross':
        criterion = CrossEntropyLoss()
    else:
        raise ValueError('Criterion not found. Available: "mse" for MSE loss, "cross" for cross-entropy loss.')

    if input_optimizer == 'sgd':
        optimizer = SGD(model, eta, momentum=0)
    elif input_optimizer == 'mom':
        optimizer = SGD(model, eta, momentum=0.9)
    elif input_optimizer == 'adam':
        optimizer = Adam(model, eta)
    else:
        raise ValueError('Optimizer not found. Available: "sgd" for SGD, "mom" for SGD with momentum, "adam" for Adam optimization.')

    losses = torch.zeros(nb_epochs)

    # Enable training mode
    model.train()

    for e in range(nb_epochs):
        loss = 0
        for b in range(0, trainX.size(0), mini_batch_size):
            # Forward pass
            out = model(trainX.narrow(0, b, mini_batch_size))

            # Compute loss
            running_loss = criterion(out, trainY.narrow(0, b, mini_batch_size))

            loss += running_loss

            # Backward pass
            model.zero_grad()
            model.backward(criterion.backward())
            optimizer.step()

        if verbose:
            if (e+1) % 10 == 0:
                print('Epoch %d/%d: loss = %.4f' % (e+1, nb_epochs, loss))
        # Collect loss data
        losses[e] = loss
    return losses
