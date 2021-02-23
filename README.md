# EE559 - Deep Learning (EPFL), Spring 2020, Project 2: "Mini deep-learning framework"

The objective of this project is to design a mini deep learning framework using only Pytorch's
tensor operations and the standard math library, hence in particular without using autograd or the
neural-network modules.

To run the project: *Python test.py* with two modes: 'train' for single training, 'trial' for N-round trial
Running time for 'train' mode: 2-3 s => running time for 'trial' mode: 20-30 s (for 10 trials)

Default network: two input units, two output units, three hidden layers of 25 units
Default loss function: MSE
Default opimizer function: SGD

Toy dataset: Training and a test set of 1,000 points sampled uniformly in [0, 1]², each with a label 0 if outside the disk of radius sqrt(1/(2pi)), and 1 inside.

The architecture of the project is described in **fig/architecture.py**. 
The *Module* class serves as a superclass for:
* Linear layers (*linear.py*)
* Activation functions (implemented: ReLU, tanh) in **activation/**
* Loss functions (implemented: MSE, cross-entropy) in **loss/**
* Dropout-like functions (implemented: inverted Dropout) (*dropout.py*)
* Sequential module (*Sequential.py*), which can contain a list of the aforementioned modules (except loss functions)

The *Optimizer* in **optim/** class serves as a superclass for optimizers (implemented: SGD (with or without momentum), Adam)

Other files and folders:
* *test.py* -> "main" of the project -> can perform a single training or a N-round trial
* *train.py* -> contains functions to train the network or to perform trials
* *utils.py* -> contains functions to generate, load and process data, creating plots
* *metrics.py* -> contains metric functions (here, only accuracy was implemented)
* **fig/** -> contains figures included in the report, as well as plotting outputs from *test.py*
