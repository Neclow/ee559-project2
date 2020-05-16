import time
import torch
from loss import *
from metrics import compute_accuracy
from optim import *
from utils import load_data


def train(model, trainX, trainY, input_criterion = 'mse', input_optimizer = 'sgd',
          nb_epochs = 250, eta=1e-3, mini_batch_size = 100, verbose = False):
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
    eta
        Learning rate
    mini_batch_size
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

def trial(net, n_trials = 30, input_criterion = 'mse', input_optimizer = 'sgd', 
          n_epochs = 250, eta = 1e-3, start_seed = 0, verbose = False, save_data = False):
    '''
    Perform a trial on a network, i.e. several rounds of training.

    Parameters
    -------
    net
        The neural network
    n_trials
        Number of trainings to perform (Default: 30)
    input_criterion
        String to choose loss function
        'mse': MSE loss
        'cross': Cross-entropy loss
    input_optimizer
        String to choose optimizer
        'sgd': SGD
        'mom': SGD with momentum (0.9)
        'adam': Adam
    n_epochs
        Number of training epochs (Default: 250)
    eta
        Learning rate
    start_seed
        Indicates from where seeds are generated.
        start_seed = 0 with 20 trials means that seeds will be 0, ..., 19
    verbose
        If true, prints final loss, training accuracy and test accuracy for each trial
    save_data
        If true, saves train and test accuracies as a tensor of size (n_trials,) in a .pt file
        Used to perform later statistical analyses (e.g. test differences of mean between configurations), if needed
        
    Returns
    -------
    all_losses
        Training losses accumulated at each epoch for each trial
    tr_accuracies
        Final train accuracy reported at the end of each trial
    te_accuracies
        Final test accuracy reported at the end of each trial
    '''

    all_losses = torch.zeros((n_trials, n_epochs))
    tr_accuracies = torch.zeros(n_trials)
    te_accuracies = torch.zeros(n_trials)
    
    for i in range(n_trials):
        # Load data
        torch.manual_seed(start_seed+i)
        trainX, trainY, testX, testY = load_data()

        # Enable training mode and reset weights
        net.train()
        net.weight_initialization()

        # Train
        start = time.time()
        tr_loss = train(net, trainX, trainY, input_criterion, 
                        input_optimizer, n_epochs, eta, verbose = False)
        print('Trial %d/%d... Training time: %.2f s' % (i+1, n_trials, time.time()-start))

        # Collect data
        all_losses[i] = tr_loss

        # Compute train and test accuracy
        net.eval() # Dropout layers will work in eval mode
        with torch.no_grad():
            tr_accuracies[i] = compute_accuracy(net, trainX, trainY)
            te_accuracies[i] = compute_accuracy(net, testX, testY)

        if verbose:
            print('Loss: %.4f, Train acc: %.4f, Test acc: %.4f' %
                  (tr_loss[-1], tr_accuracies[i], te_accuracies[i]))

    # Print trial results
    print('Train accuracy - mean: %.4f, std: %.4f, median: %.4f' %
         (tr_accuracies.mean(), tr_accuracies.std(), tr_accuracies.median()))
    print('Test accuracy - mean: %.4f, std: %.4f, median: %.4f' %
         (te_accuracies.mean(), te_accuracies.std(), te_accuracies.median()))
    
    if save_data:
        torch.save(tr_accuracies, f'data/train_{input_optimizer}_{input_criterion}_{len(net)}.pt')
        torch.save(te_accuracies, f'data/test_{input_optimizer}_{input_criterion}_{len(net)}.pt')