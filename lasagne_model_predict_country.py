import lasagne
import numpy as np
import theano
import theano.tensor as T
import time

def simple_cnn(C, W, H, num_classes, num_filters=32, filter_width=5, pool_width=2, stride=1, pad='valid', hidden_size=256, dropout=0.0, inputVar = None):
  '''

  Two convolution + pooling stages and a fully-connected hidden layer in front of the output layer.
  INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC
  '''

  print 'Building a simple CNN...'

  # Input layer
  l_in = lasagne.layers.InputLayer(shape=(None, C, W, H), input_var=inputVar)  

  # Convolutional layer with num_filters kernels of size filter_width x filter_width. 
  network = lasagne.layers.Conv2DLayer(
          l_in, num_filters=num_filters, filter_size=filter_width, stride=stride, pad=pad,
          nonlinearity=lasagne.nonlinearities.rectify,
          W=lasagne.init.GlorotUniform())

  # Max-pooling layer of factor pool_width in both dimensions:
  network = lasagne.layers.MaxPool2DLayer(network, pool_size=pool_width)

  # Another convolution and pooling:
  network = lasagne.layers.Conv2DLayer(
          network, num_filters=num_filters, filter_size=filter_width, stride=stride, pad=pad,
          nonlinearity=lasagne.nonlinearities.rectify,)
  network = lasagne.layers.MaxPool2DLayer(network, pool_size=(pool_width, pool_width))

  # A fully-connected layer of hidden_size units with dropout on its inputs:
  network = lasagne.layers.DenseLayer(
          lasagne.layers.dropout(network, p=dropout),
          num_units=hidden_size,
          nonlinearity=lasagne.nonlinearities.rectify)

   # And, finally, the num_classes-unit output layer with dropout on its inputs:
  l_out = lasagne.layers.DenseLayer(
          lasagne.layers.dropout(network, p=dropout),
          num_units=num_classes,
          nonlinearity=lasagne.nonlinearities.softmax)

  return l_in, l_out


def complex_cnn(C, W, H, num_classes, num_filters=32, filter_width=5, pool_width=2, stride=1, pad=0, hidden_size=256, dropout=0.0, inputVar = None):
  '''

  Two convolution + pooling stages and a fully-connected hidden layer in front of the output layer.
  INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC
  '''

  print 'Building a complex CNN...'

  # Input layer
  l_in = lasagne.layers.InputLayer(shape=(None, C, W, H), input_var=inputVar)  

  network = l_in
  for _ in xrange(3):
    # 2 Conv - ReLu layers
    network = lasagne.layers.Conv2DLayer(
          network, num_filters=num_filters, filter_size=filter_width, stride=stride, pad=pad,
          nonlinearity=lasagne.nonlinearities.rectify,
          W=lasagne.init.GlorotUniform())

    network = lasagne.layers.Conv2DLayer(
            l_in, num_filters=num_filters, filter_size=filter_width, stride=stride, pad=pad,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Max-pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=pool_width)

  for _ in xrange(2):
    # FC - ReLu
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=dropout),
            num_units=hidden_size,
            nonlinearity=lasagne.nonlinearities.rectify)

   # FC
  l_out = lasagne.layers.DenseLayer(
          lasagne.layers.dropout(network, p=dropout),
          num_units=num_classes,
          nonlinearity=lasagne.nonlinearities.softmax)

  return l_in, l_out


###
# Using example from https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
###

def build_cnn(C, W, H, num_classes, num_filters=32, filter_width=5, pool_width=2, stride=1, pad=0, hidden_size=256, dropout=0.0, inputVar = None, cnn_architecture="simple_cnn"):
  '''
  Builds and returns CNN model

  Args:
    num_classes: number of classes being predicted (Used as num_units in final output layer)
  '''

  if cnn_architecture == "simple_cnn":
    l_in, l_out = simple_cnn(C, W, H, num_classes, num_filters, filter_width, pool_width, stride, pad, hidden_size, dropout, inputVar)
  elif cnn_architecture == "complex_cnn":
    l_in, l_out = complex_cnn(C, W, H, num_classes, num_filters, filter_width, pool_width, stride, pad, hidden_size, dropout, inputVar)
  else:
    print 'Given CNN architecture does not exist.'

  return l_in, l_out

  
