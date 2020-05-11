import math
import torch
import matplotlib.pyplot as plt


def load_data(plotting=False):
    r2 = 1/(2*math.pi)
    trainX = torch.FloatTensor(1000, 2).uniform_(0, 1)
    testX = torch.FloatTensor(1000, 2).uniform_(0, 1)
    center = torch.FloatTensor([0.5, 0.5]) # Center of the disc
    trainY = to_one_hot((trainX.sub(center).pow(2).sum(axis=1) < r2).long())
    testY = to_one_hot((testX.sub(center).pow(2).sum(axis=1) < r2).long())
    
    mean_ = trainX.mean()
    std_ = trainX.std()
    
    if plotting:
        plot_dataset(trainX, trainY, 'train')
        plot_dataset(testX, testY, 'test')
    
    return standardize(trainX, mean_, std_), trainY, standardize(testX, mean_, std_), testY

def to_one_hot(y):
    Y = torch.zeros((len(y), len(y.unique())))
    
    Y[range(len(y)), y] = 1
    
    return Y

def standardize(x, mean, std):
    return x.sub_(mean).div_(std)

def plot_dataset(data, labels, name):
    plt.scatter(data[:,0], data[:,1], c=['r' if p==0 else 'b' for p in labels.argmax(1)])
    #plt.show()
    
    fname = 'fig/' + name + '.png'

    print(f'Plot of {name} data saved under {fname} \n')
    
    plt.savefig(fname)

def train_visualization(net, losses, testX, testY):
    # Plot losses
    plt.figure()
    
    plt.plot(range(1,len(losses)+1), losses, 'k--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    #plt.show()
    loss_fname = 'fig/loss.png'
    print(f'Plot of training loss saved under {loss_fname}')
    plt.savefig(loss_fname)
    
    # Get test predictions for future plots
    pred = net(testX).argmax(1)

    b00 = (pred == 0) & (testY.argmax(1) == 0) # Predicted as 0 and true class is 0
    b01 = (pred == 0) & (testY.argmax(1) == 1) # Predicted as 0 and true class is 1
    b10 = (pred == 1) & (testY.argmax(1) == 0) # Predicted as 1 and true class is 0
    b11 = (pred == 1) & (testY.argmax(1) == 1) # Predicted as 1 and true class is 1
    
    # Plot correlation matrix
    plt.figure()
    corr = torch.tensor([[b00.sum(), b10.sum()], [b01.sum(), b11.sum()]])
    plt.matshow(corr, cmap='tab10')

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(i, j, '{:d}'.format(corr[i,j]), ha='center', va='center')

    plt.xlabel('True Class')
    plt.ylabel('Predicted class')
    #plt.show()
    corr_fname = 'fig/corrmat.png'
    print(f'Plot of correlation matrix saved under {corr_fname}')
    plt.savefig(corr_fname)
    
    # Plotting scattering of correctly (resp. falsely) predicted values 
    plt.figure()
    correct0 = torch.nonzero(b00==True)
    correct1 = torch.nonzero(b11==True)
    plt.scatter(testX[correct0,0], testX[correct0, 1], c='r', alpha=0.2, label='class 0')
    plt.scatter(testX[correct1,0], testX[correct1, 1], c='b', alpha=0.2, label='class 1')
    errors = torch.cat((torch.nonzero(b01==True), torch.nonzero(b10 == True)), dim=0)
    plt.scatter(testX[errors,0], testX[errors, 1], c='k', label='errors')
    plt.legend()
    #plt.show()
    pred_fname = 'fig/predictions.png'
    print(f'Plot of test predictions saved under {pred_fname}')
    plt.savefig(pred_fname)