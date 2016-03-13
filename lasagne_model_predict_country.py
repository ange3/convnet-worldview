import lasagne
import numpy as np
import theano
import theano.tensor as T
import time
import data_utils

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

def _build_cnn(C, W, H, num_classes, num_filters=32, filter_width=5, pool_width=2, stride=1, pad=0, hidden_size=256, dropout=0.0, inputVar = None, cnn_architecture="simple_cnn"):
  '''
  Wrapper that builds and returns CNN model with specified architecture.

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


def main_create_model(C, W, H, NUM_CLASSES, cnn_architecture="simple_cnn", num_filters=32, filter_width=5, pool_width=2, stride=1, pad=0, hidden_size=256, dropout=0.0, use_optimizer = "nesterov_momentum", learning_rate=1e-2, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-08):
  '''
  Builds the CNN model:
    (1) Builds the network
    (2) Calculates loss and accuracy expression
    (3) Compiles train function (assigning appropriate update) and val function

  Returns:
    train function
    val function
    l_out (final layer of network)
  '''
  # Prepare Theano variables for inputs and targets
  input_var = T.tensor4('inputs')
  target_var = T.ivector('targets')

  print('Building network...')

  # Create neural network model
  l_in, l_out = _build_cnn(C, W, H, NUM_CLASSES, cnn_architecture=cnn_architecture, num_filters=num_filters, filter_width=filter_width, pool_width=pool_width, stride=stride, pad=pad, hidden_size=hidden_size, dropout=dropout, inputVar = input_var)

  print('Compiling functions...')

  # Create a loss expression for training, i.e., a scalar objective we want
  # to minimize (for our multi-class problem, it is the cross-entropy loss):
  prediction = lasagne.layers.get_output(l_out)
  loss = T.nnet.categorical_crossentropy(prediction, target_var)
  # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
  loss = loss.mean()
  acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),  # prediction (class with max prob) --> if equivalent to answer --> mean of all samples (to get accuracy)
                    dtype=theano.config.floatX)

  # Return predictions in a function
  pred_fn = theano.function([l_in.input_var], prediction)

  # Create update expressions for training, i.e., how to modify the
  # parameters at each training step. Here, we'll use Stochastic Gradient
  # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
  params = lasagne.layers.get_all_params(l_out, trainable=True)
  if use_optimizer == "nesterov_momentum":
      updates = lasagne.updates.nesterov_momentum(
          loss, params, learning_rate=learning_rate, momentum=momentum)
  elif use_optimizer == "adam":
      updates = lasagne.updates.adam(
          loss, params, learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
  else:   # default is nesterov momentum
      updates = lasagne.updates.nesterov_momentum(
          loss, params, learning_rate=learning_rate, momentum=momentum)

  print 'Using Update:', use_optimizer
      
  # Create a loss expression for validation/testing. The crucial difference
  # here is that we do a deterministic forward pass through the network,
  # disabling dropout layers.
  test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
  # test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
  test_loss = T.nnet.categorical_crossentropy(test_prediction, target_var)

  test_loss = test_loss.mean()
  # As a bonus, also create an expression for the classification accuracy:
  test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                    dtype=theano.config.floatX)

   # Compile a function performing a training step on a mini-batch (by giving
  # the updates dictionary) and returning the corresponding training loss:
  train_fn = theano.function([input_var, target_var], [loss, acc], updates=updates)
  # train_fn = theano.function([input_var, target_var], loss, updates=updates)

  # Compile a second function computing the validation loss and accuracy:
  val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

  print('Compiling Finished!')

  return train_fn, val_fn, pred_fn, l_out


def train(num_epochs, batchsize, num_train, num_val, use_optimizer, train_fn, val_fn, X_train, y_train, X_val, y_val, record_per_iter = True, save_results_to_file=None):
  '''
  Trains CNN model using given train and val functions
  Args:
    Training Params:  num_epochs, batchsize, num_train
    Functions:        train_fn, val_fn
    Data:             X_train, y_train

  Returns: Lists of loss (error) and accuracy values by iteration and by epoch

  Can save epoch results to file.
  '''
  print ( 'Training on: {} epochs of batch size {} with num training samples {}'.format(num_epochs,batchsize,num_train) )
  print ('Using optimizer: {}'.format(use_optimizer))
  print ("Validation Size: {}".format(num_val) )
  print("Starting training...")

  # Data structures to store error and accuracy values
  train_err_list = []
  val_err_list = []
  train_acc_list = []
  val_acc_list = []

  epochs_train_err_list = []
  epochs_train_acc_list = []
  epochs_val_err_list = []
  epochs_val_acc_list = []

  if save_results_to_file:
    results_text_all = []

  # We iterate over epochs:
  for epoch in range(num_epochs):
      # 1) In each epoch, we do a full pass over the training data:
      train_err = 0
      train_acc = 0
      train_batches = 0
      start_time = time.time()
      
      # Each batch is 1 iteration
      num_iters = 0
      for batch in data_utils.iterate_minibatches(X_train, y_train, batchsize, shuffle=True):
          # For testing, limit num iterations
          # if num_iters >= 2:
          #     break
          inputs, targets = batch
          iter_train_err, iter_train_acc = train_fn(inputs, targets)

          # Can use this function to take out slice prediction at last step
          # prediction = pred_fn(inputs)  

          train_err += iter_train_err
          train_acc += iter_train_acc
          train_batches += 1
          
          # Run validation on entire val data set after each iteration, not iterating over mini batches
          val_err, val_acc = val_fn(X_val, y_val)
          
          if record_per_iter:
              train_err_list.append(iter_train_err)
              train_acc_list.append(iter_train_acc)
              val_err_list.append(val_err)
              val_acc_list.append(val_acc)
          
          iter_results = "Ep {} \titer {}  \tloss {:.5f}, train acc {:.2f}, val acc {:.2f}".format(epoch, num_iters, float(iter_train_err), iter_train_acc * 100, val_acc *100)
          print iter_results
          if save_results_to_file:
            results_text_all.append(iter_results)
          num_iters += 1

      # Then we print the results for this epoch:
      
      epoch_train_err = train_err / train_batches
      epoch_train_acc = train_acc / train_batches * 100
      epoch_val_err = val_err / 1
      epoch_val_acc = val_acc / 1 * 100
      
      epoch_results = "Epoch {} of {} took {:.3f}s\n".format(epoch + 1, num_epochs, time.time() - start_time)
      epoch_results += "  training loss:\t\t{:.6f}\n".format(epoch_train_err)
      epoch_results += "  training accuracy:\t\t{:.2f} %\n".format(epoch_train_acc)
      epoch_results += "  validation loss:\t\t{:.6f}\n".format(epoch_val_err)
      epoch_results += "  validation accuracy:\t\t{:.2f} %".format(epoch_val_acc)
      print epoch_results

      if save_results_to_file:
        results_text_all.append(epoch_results)
      
      # Record loss and accuracy per epoch as well
      epochs_train_err_list.append(epoch_train_err)
      epochs_train_acc_list.append(epoch_train_acc)
      epochs_val_err_list.append(epoch_val_err)
      epochs_val_acc_list.append(epoch_val_acc)

  print('Training finished!')

  # Save training results to file
  if save_results_to_file:
    print ('Saving training results to file...')
    f = open (save_results_to_file, 'wb')
    for text in results_text_all:
      f.write(text)
      f.write("\n")
    print ('Saving results to file finished!')

  return train_err_list, train_acc_list, val_err_list, val_acc_list, epochs_train_err_list, epochs_train_acc_list, epochs_val_err_list, epochs_val_acc_list