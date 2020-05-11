import torch
import math
import time
import matplotlib.pyplot as plt
from linear import Linear
from Sequential import Sequential
from module import Module
from activation import *
from loss import *
from optim import *
from utils import load_data, train_visualization
from train import train
from metrics import compute_accuracy


def main():
    print('Default run: single training with default net, MSE loss and SGD.')
    print('To visualize data, change flag "plot_data" to True.')
    print('To visualize training loss and predictions, change flag "plot_training" to True.')
    print('Available criterion: "mse" (default), "cross" for cross-entropy loss')
    print('Available optimizers: "sgd" (default), "mom" for SGD + momentum, "adam" for Adam optimization')
    
    print('Recommended learning rates: ')
    print('SGD: 1e-1 with MSE loss, 1e-2 with Cross-Entropy loss')
    print('SGD + Momentum: 1e-2 with MSE loss, 1e-4 with Cross-Entropy loss')
    print('Adam: 1e-3 \n')

    seed = 42
    torch.manual_seed(seed)
    plot_data = True
    plot_training = True
    trainX, trainY, testX, testY = load_data(plotting=plot_data)
    
    net = Sequential([Linear(2, 25),
                      ReLU(),
                      Linear(25, 25),
                      ReLU(),
                      Linear(25, 25),
                      ReLU(),
                      Linear(25, 25),
                      ReLU(),
                      Linear(25, 2)])
    
    criterion = 'mse'
    optimizer = 'sgd'
    eta = 1e-1
    
    print('Model: ')
    print(net)
    
    time.sleep(2)
    
    losses = train(net, trainX, trainY, input_criterion=criterion, input_optimizer=optimizer, eta=eta, verbose=True)
    
    print('Train accuracy: %.4f' % compute_accuracy(net, trainX, trainY))
    print('Test accuracy: %.4f' % compute_accuracy(net, testX, testY))
    print()
    
    if plot_training:
        train_visualization(net, losses, testX, testY)

if __name__ == "__main__":
    main()