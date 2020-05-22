# Python libraries
import torch
import math
import time
import matplotlib.pyplot as plt


# Neural network classes
from Sequential import Sequential
from linear import Linear
from activation import *
from loss import *
from optim import *
from dropout import InvertedDropout

# Other classes useful for deep-learning
from utils import load_data, train_visualization
from train import train, trial
from metrics import compute_accuracy


def run_train(net, criterion, optimizer, eta, plot_data = False, plot_training = False, seed = 42):
    torch.manual_seed(seed)

    # Load data
    trainX, trainY, testX, testY = load_data(plotting=plot_data)

    # Initialize weights
    net.weight_initialization()

    time.sleep(2)

    # Train model
    losses = train(net, trainX, trainY, input_criterion=criterion, input_optimizer=optimizer,
                   eta=eta, verbose=True)

    # Compute accuracy
    net.eval()
    print('Train accuracy: %.4f' % compute_accuracy(net, trainX, trainY))
    print('Test accuracy: %.4f \n' % compute_accuracy(net, testX, testY))

    # Create if plots (if flag is True)
    if plot_training:
        train_visualization(net, losses, testX, testY)


def main():
    '''
    Main function.
    Runs a single training, or 10 trials with default model, loss function and optimizer
    '''

    print('Default run: single training with default net, MSE loss and SGD.')
    print('Available activation functions: ReLU, tanh.')
    print('Available criteria: "mse" for MSE loss (default), "cross" for cross-entropy loss')
    print('Available optimizers: "sgd" for SGD (default), "mom" for SGD + momentum, "adam" for Adam optimization')

    print('Recommended learning rates: ')
    print('SGD: 1e-2 with MSE loss, 5e-4 with Cross-Entropy loss')
    print('SGD + Momentum: 1e-3 with MSE loss, 1e-4 with Cross-Entropy loss')
    print('Adam: 1e-3 \n')


    # Load default model
    net = Sequential([Linear(2, 25),
                      ReLU(),
                      Linear(25, 25),
                      ReLU(),
                      Linear(25, 25),
                      ReLU(),
                      Linear(25, 2)])

    print(net)

    # Load default criterion and optimizer, with corresponding LR
    criterion = 'mse'
    optimizer = 'sgd'
    eta = 1e-2

    # Running mode: 'train' for single training, 'trial' for several trainings
    mode = 'train'
    #mode = 'trial'

    print(f'\n Selected mode: {mode} \n')
    time.sleep(1)

    if mode == 'train':
        print('To visualize data, change flag "plot_data" to True.')
        print('To visualize training loss and predictions, change flag "plot_training" to True.')
        plot_data = True
        plot_training = True
        run_train(net, criterion, optimizer, eta, plot_data = plot_data, plot_training = plot_training)
    elif mode == 'trial':
        n_trials = 10
        trial(net, n_trials = n_trials, input_criterion = criterion,
              input_optimizer = optimizer, eta = eta, verbose = True)
    else:
        raise ValueError('Running mode not found. Try "train" for simple train, "trial" for full trial.')

if __name__ == "__main__":
    main()
