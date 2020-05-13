def compute_accuracy(net, input_, target):
    '''
    Compute accuracy of trained model

    Parameters
    -------
    net
        Trained model
    input_
        Predicted class
    target
        True classes

    Returns
    -------
    tensor
        Computed accuracy
    '''
    # Testing mode
    net.eval()
    return (net(input_).argmax(1) == target.argmax(1)).float().mean()
