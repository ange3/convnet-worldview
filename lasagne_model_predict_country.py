import lasagne
import numpy as np
import theano
import theano.tensor as T
import time

###
# Using example from https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
###

def build_cnn(C, W, H, num_classes, num_filters=32, filter_width=5, pool_width=2, hidden_size=256, dropout=0.0, inputVar = None):
  '''
  Builds and returns CNN model
  Two convolution + pooling stages and a fully-connected hidden layer in front of the output layer.

  Args:
    num_classes: number of classes being predicted (Used as num_units in final output layer)
  '''

  # Input layer
  l_in = lasagne.layers.InputLayer(shape=(None, C, W, H), input_var=inputVar)  

  # Convolutional layer with 32 kernels of size 5x5. Strided and padded
  # convolutions are supported as well; see the docstring.
  network = lasagne.layers.Conv2DLayer(
          l_in, num_filters=num_filters, filter_size=(filter_width, filter_width),
          nonlinearity=lasagne.nonlinearities.rectify,
          W=lasagne.init.GlorotUniform())

  # Max-pooling layer of factor 2 in both dimensions:
  network = lasagne.layers.MaxPool2DLayer(network, pool_size=(pool_width, pool_width))

  # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
  network = lasagne.layers.Conv2DLayer(
          network, num_filters=num_filters, filter_size=(filter_width, filter_width),
          nonlinearity=lasagne.nonlinearities.rectify)
  network = lasagne.layers.MaxPool2DLayer(network, pool_size=(pool_width, pool_width))

  # A fully-connected layer of 256 units with dropout on its inputs:
  network = lasagne.layers.DenseLayer(
          lasagne.layers.dropout(network, p=dropout),
          num_units=hidden_size,
          nonlinearity=lasagne.nonlinearities.rectify)

   # And, finally, the 5-unit output layer with dropout on its inputs:
  l_out = lasagne.layers.DenseLayer(
          lasagne.layers.dropout(network, p=dropout),
          num_units=num_classes,
          nonlinearity=lasagne.nonlinearities.softmax)

  return l_in, l_out

