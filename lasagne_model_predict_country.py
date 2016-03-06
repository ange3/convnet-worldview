import lasagne
import numpy as np
import theano
import theano.tensor as T
import time

###
# Using example from https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
###

def hello_world():
  print 'hello world!'

def build_cnn(num_classes, inputVar = None):
  '''
  Builds and returns CNN model
  Two convolution + pooling stages and a fully-connected hidden layer in front of the output layer.

  Args:
    num_classes: number of classes being predicted (Used as num_units in final output layer)
  '''

  # Input layer
  network = lasagne.layers.InputLayer(shape=(None, 3, 48, 32), input_var=inputVar)  

  # Convolutional layer with 32 kernels of size 5x5. Strided and padded
  # convolutions are supported as well; see the docstring.
  network = lasagne.layers.Conv2DLayer(
          network, num_filters=32, filter_size=(5, 5),
          nonlinearity=lasagne.nonlinearities.rectify,
          W=lasagne.init.GlorotUniform())

  # Max-pooling layer of factor 2 in both dimensions:
  network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

  # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
  network = lasagne.layers.Conv2DLayer(
          network, num_filters=32, filter_size=(5, 5),
          nonlinearity=lasagne.nonlinearities.rectify)
  network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

  # A fully-connected layer of 256 units with 50% dropout on its inputs:
  network = lasagne.layers.DenseLayer(
          lasagne.layers.dropout(network, p=.5),
          num_units=256,
          nonlinearity=lasagne.nonlinearities.rectify)

   # And, finally, the 10-unit output layer with 50% dropout on its inputs:
  network = lasagne.layers.DenseLayer(
          lasagne.layers.dropout(network, p=.5),
          num_units=num_classes,
          nonlinearity=lasagne.nonlinearities.softmax)

  return network

