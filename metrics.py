def compute_accuracy(net, input_, target):
    return (net(input_).argmax(1) == target.argmax(1)).float().mean()