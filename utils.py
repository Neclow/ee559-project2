import math

import torch

def load_data():
    r2 = 1/(2*math.pi)
    trainX = torch.FloatTensor(1000, 2).uniform_(0, 1)
    testX = torch.FloatTensor(1000, 2).uniform_(0, 1)
    center = torch.FloatTensor([0.5, 0.5]) # Center of the disc
    trainY = (trainX.sub(center).pow(2).sum(axis=1) < r2).long()
    testY = (testX.sub(center).pow(2).sum(axis=1) < r2).long()
    
    mean_ = trainX.mean()
    std_ = trainX.std()
    
    
    #return trainX, to_one_hot(trainY), testX, to_one_hot(testY)
    return standardize(trainX, mean_, std_), to_one_hot(trainY), standardize(testX, mean_, std_), to_one_hot(testY)

def to_one_hot(y):
    Y = torch.zeros((len(y), len(y.unique())))
    
    Y[range(len(y)), y] = 1
    
    return Y

def standardize(x, mean, std):
    return x.sub_(mean).div_(std)