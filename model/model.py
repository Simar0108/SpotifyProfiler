"""network3.py
~~~~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

Written for Theano 0.6 and 0.7, needs some changes for more recent
versions of Theano.

"""

#### Libraries
# Standard library
import pickle # for loading compressed data
import gzip # for loading compressed data

# Third-party libraries
import numpy as np
import theano # symbolic math library that allows us to define and train neural networks by compiling math expressions into executable code
import theano.tensor as T # tensor library that allows us to define and manipulate tensors (multi-dimensional arrays)
from theano.tensor.nnet import conv # convolutional layer
from theano.tensor.nnet import softmax # softmax layer
from theano.tensor import shared_randomstreams # shared random streams for dropout

# theano can automatically compute gradients and optimize mathematical expressions

# Compatibility for pooling across Theano variants
try:
    from theano.tensor.signal import downsample as _downsample
    max_pool_2d = _downsample.max_pool_2d
except Exception:
    from theano.tensor.signal import pool as _pool
    max_pool_2d = _pool.pool_2d

# Activation functions for neurons
def linear(z): return z # linear activatino function: output is equal to input
def ReLU(z): return T.maximum(0.0, z) # Rectified Linear Unit - max(0,z). Makes all negative values 0 while keeping positive values unchanged. Helps with the vanishing gradient problem.
from theano.tensor.nnet import sigmoid # sigmoid activation function: output is between 0 and 1
from theano.tensor import tanh # hyperbolic tangent activation function: output is between -1 and 1


#### Constants. GPU is set to False by default. If you want to use a GPU, set GPU to True. GPU is much faster than CPU.
GPU = False
if GPU:
    print("Trying to run under a GPU. If this is not desired, then modify network3.py to set the GPU flag to False.")
    try:
        theano.config.device = 'gpu'
    except Exception:
        pass  # it's already set
    theano.config.floatX = 'float32'
else:
    print("Running with a CPU. If this is not desired, then modify network3.py to set the GPU flag to True.")



#### Load the MNIST data
def load_data_shared(filename="../data/mnist.pkl.gz"):  # loads and converts data to theano shared variables
    f = gzip.open(filename, 'rb') # opens the file in binary read mode and decompresses it
    training_data, validation_data, test_data = pickle.load(f) # loads the data into 3 datasets containing tuples of (features/input, labels/target)
    f.close() # closes the file
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available. This is a performance optimization.
        """
        shared_x = theano.shared( # creates a shared variable that can be used by the GPU
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True) # converts the data (features) to a numpy array and sets the data type to float32
        shared_y = theano.shared( # creates a shared variable that can be used by the GPU
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True) # converts the data (labels) to a numpy array and sets the data type to float32
        return shared_x, T.cast(shared_y, "int32") # casts the labels to integers. Labels are discrete classes, not continuous values.
    return [shared(training_data), shared(validation_data), shared(test_data)] # returns the 3 datasets as a list of tuples of (features, labels)

#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers # stores the list of layers in the network
        self.mini_batch_size = mini_batch_size # size of the mini-batches for training
        self.params = [param for layer in self.layers for param in layer.params] # stores all the parameters of all the layers in the network
        self.x = T.matrix("x") # symbolic matric for input features
        self.y = T.ivector("y") # symbolic vector for target labels
        init_layer = self.layers[0] # initializes the first layer
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size) # sets the input for the first layer
        for j in range(1, len(self.layers)): # iterates over the remaining layers. Each layer gets its input from the previous layer. This is a feedforward network.
            prev_layer, layer  = self.layers[j-1], self.layers[j] # gets the previous and current layers
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size) # sets the input for the current layer
        self.output = self.layers[-1].output # stores the output of the last layer
        self.output_dropout = self.layers[-1].output_dropout # stores the output of the last layer with dropout. Dropout is a regularization technique that randomly drops out some of the neurons in the network during training to prevent overfitting.

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent.""" #splitting the datasets into features and labels
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers]) # L2 regularization: penalizes large weights. This helps prevent overfitting.
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches # cost function is the negative log-likelihood of the correct class plus a regularization term to prevent overfitting. lmbda is the regularization parameter.
        grads = T.grad(cost, self.params) # computes the gradients of the cost function with respect to the parameters
        updates = [(param, param-eta*grad) # updates the parameters using the gradients and the learning rate. eta is the learning rate.
                   for param, grad in zip(self.params, grads)] # zip combines the parameters and gradients into a list of tuples

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function( # defines a function that trains a mini-batch
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function( # defines a function that computes the accuracy in validation mini-batches
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function( # defines a function that computes the accuracy in test mini-batches
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function( # defines a function that computes the predictions in test mini-batches
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        # Do the actual training
        best_validation_accuracy = 0.0 # initializes the best validation accuracy to 0
        for epoch in range(epochs): # iterates over the epochs
            for minibatch_index in range(num_training_batches): # iterates over the mini-batches
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0: # prints the training progress every 1000 iterations
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index) # trains a mini-batch
                if (iteration+1) % num_training_batches == 0: # checks if the end of the epoch is reached
                    validation_accuracy = np.mean( # computes the average accuracy in validation mini-batches
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format( # prints the validation accuracy
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy: # checks if the validation accuracy is the best so far
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy # updates the best validation accuracy
                        best_iteration = iteration # updates the best iteration
                        if test_data: # checks if the test data is available
                            test_accuracy = np.mean( # computes the average accuracy in test mini-batches
                                [test_mb_accuracy(j) for j in range(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format( # prints the test accuracy
                                test_accuracy))
                                # overall, we have 60,000 samples in the training set, 10,000 in the validation set, and 10,000 in the test set. We have mini batches of 100 samples, so we do 600 weight updates per epoch, with each update using a different randome subset of 100 samples.
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """

        ''' It basically is having each neuron only look at a small patch of the input image and learn a feature from that patch.
        The filter_shape is the shape of the filter, which is the number of filters, the number of input feature maps, the filter height, and the filter width.
        The image_shape is the shape of the input image, which is the mini-batch size, the number of input feature maps, the image height, and the image width.
        The poolsize is the size of the pooling window, which is the y and x pooling sizes.
        The activation_fn is the activation function to be used in the layer.
        '''

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize)) # number of output features is the number of filters times the number of input feature maps divided by the pooling size.
        # each filter produces one output feature map, pooling reduces the spatial dimensions of the feature maps, and so n_out tell us how many parameters we need to learn for each filter.
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size): 
        self.inpt = inpt.reshape(self.image_shape) # reshapes the input to the shape of the image to 4D tensor (mini_batch_size, num_input_feature_maps, image_height, image_width)
        conv_out = conv.conv2d( # performs a 2D convolution
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape, # input is the input tensor, filters is the weights tensor, filter_shape is the shape of the filters
            image_shape=self.image_shape) # image_shape is the shape of the input image
        pooled_out = max_pool_2d( # performs a 2D max pooling
            conv_out, self.poolsize, ignore_border=True) # conv_out is the output of the convolution, poolsize is the size of the pooling window, ignore_border=True means that we ignore the border pixels
        self.output = self.activation_fn( # applies the activation function to the output
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')) # adds the bias term to the output
        self.output_dropout = self.output # no dropout in the convolutional layers because we want to keep all the information from the convolutional layer.

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)), # initializes the weights to a normal distribution with mean 0 and standard deviation 1/sqrt(n_out).
                    # Called Xavier initialization. This helps prevent the vanishing gradient problem (if the weights are too large, the gradient will be too small and the network will not learn).
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)), # initializes the biases to a normal distribution with mean 0 and standard deviation 1.
                       dtype=theano.config.floatX), # sets the data type to float32
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in)) # reshapes the input to the shape of the input layer
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b) # applies the activation function to the output, and applies dropout to the input to scale down weights.
        self.y_out = T.argmax(self.output, axis=1) # computes the predicted class
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout) # applies dropout to the input to scale down weights. This is the normal forward pass.
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b) # applies the activation function to the output, and applies dropout to the input to scale down weights. This is the forward pass with dropout. Used during testing.

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out)) # computes the accuracy of the predictions. 
        # T.eq(y, self.y_out) is a boolean array of 1s and 0s, where 1s are correct predictions and 0s are incorrect predictions. T.mean computes the average of the boolean array.

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases. We initialize the weights to 0 and the biases to 0 because we want to start with a clean slate, not biases that are too large or too small.
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b) # converts raw scores to probabilities. Does this by finding exponential of the raw scores and then normalizing them.
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        # Takes the predicted probabilities for the correct class and takes the negative log of it (since we want to minimize cost -> maximize log-likelihood). This is the negative log-likelihood cost. Averages over all the samples in the mini-batch.
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out)) # computes the accuracy of the predictions. 
        # T.eq(y, self.y_out) is a boolean array of 1s and 0s, where 1s are correct predictions and 0s are incorrect predictions. T.mean computes the average of the boolean array.


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0] # returns the number of samples in the dataset, e.g. the number of features in the dataset.

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape) # creates a mask of 1s and 0s, where 1s are kept and 0s are dropped out. p_dropout is the probability of a neuron being dropped out.
    return layer*T.cast(mask, theano.config.floatX) # multiplies the input by the mask to drop out the neurons. So it essentially scales down the weights of the neurons that are dropped out.