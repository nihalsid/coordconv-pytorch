# CoordConv Pytorch Lightning

CoordConv regression task from the paper "An intriguing failing of convolutional neural networks and the CoordConv solution". 

### Files

`create_data`: scripts for creating the NotSoClvr dataset and Floating MNIST data

`dataset`: dataset class; see the test case for how to use

`model`: architectures proposed in the paper

`train`: lightning module for training; see util/arguments for arg specification

`util`: argument and logger methods/classes

### Example Command

`python train.py --dataset floating_mnist_quadrant --coords coord --experiment sample_experiment --max_epochs 50`

The tensorflow logs and visualizations are saved in `runs` directory in the root folder
