

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
# from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


"""
MNIST classification using TensorFlow 2.0
"""


# Load and prepare the MNIST dataset-
mnist = tf.keras.datasets.mnist

# type(mnist)
# module

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# type(X_train), type(y_train), type(X_test), type(y_test)
# (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)


# Normalize and convert samples from integers to floating-point numbers-
X_train, X_test = X_train / 255.0, X_test / 255.0

y_train = tf.cast(y_train, dtype=tf.float32)
y_test = tf.cast(y_test, dtype=tf.float32)

print("\nShapes of training and testing sets are:")
print("X_train.shape = {0}, y_train.shape = {1}, X_test.shape = {2} & y_test.shape = {3}\n".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
# Shapes of training and testing sets are:
# X_train.shape = (60000, 28, 28), y_train.shape = (60000,), X_test.shape = (10000, 28, 28) & y_test.shape = (10000,)


# One-Hot encode target variables-

y_train_ohe = tf.one_hot(tf.cast(y_train, dtype=tf.int32), depth = 10, dtype=tf.float32)
y_test_ohe = tf.one_hot(tf.cast(y_test, dtype=tf.int32), depth = 10, dtype=tf.float32)


# Reshape 'X_train' and 'X_test' to be a single arrays of inputs-
X_train_reshaped = X_train.reshape(X_train.shape[0], 28 * 28)
X_test_reshaped = X_test.reshape(X_test.shape[0], 28 * 28)

# Cast from tf.float64 to tf.float32-
X_train_reshaped = tf.cast(X_train_reshaped, dtype=tf.float32)
X_test_reshaped = tf.cast(X_test_reshaped, dtype=tf.float32)

print("\nShapes of reshaped training and testing sets are: ")
print("X_train_reshaped.shape = {0}, X_test_reshaped.shape = {1}\n".format(X_train_reshaped.shape, X_test_reshaped.shape))
# Shapes of reshaped training and testing sets are: 
# X_train_reshaped.shape = (60000, 784), X_test_reshaped.shape = (10000, 784)

print("\nShapes of training and testing labels AFTER one-hot encoding:")
print("y_train_ohe.shape = {0} & y_test_ohe.shape = {1}\n".format(y_train_ohe.shape, y_test_ohe.shape))
# Shapes of training and testing labels AFTER one-hot encoding:
# y_train_ohe.shape = (60000, 10) & y_test_ohe.shape = (10000, 10)




def relu(x):
	'''
	Function to calculate ReLU for
	given 'x'
	'''
	# return np.maximum(x, 0)
	return tf.cast(tf.math.maximum(x, 0), dtype = tf.float32)


def relu_derivative(x):
	'''
	Function to calculate derivative
	of ReLU
	'''
	# return np.where(x <= 0, 0, 1)
	# return tf.where(x <=0, 0, 1)
	return tf.cast(tf.where(x <=0, 0, 1), dtype=tf.float32)


def softmax_stable(z):
    '''
    Function to compute softmax activation function.
    Numerically stable
    '''
    # First cast 'z' to floating type-
    z = tf.cast(z, dtype = tf.float32)

    # Get largest element in 'z'-
    largest = tf.math.reduce_max(z)

    # Raise each value to exp('z - largest')-
    z_exp = tf.math.exp(z - largest)

    # Compute softmax activation values-
    s = z_exp / tf.math.reduce_sum(z_exp)

    return s


def initialize_parameters(input_layer, hidden_layer, output_layer):
	'''
	Function to initialize parameters for a neural network with-
	'input_layer' number of neurons in input layer
	'hidden_layer' number of neurons in hidden layer
	'output_layer' neuron in output layer [output layer has one neuron]

	Initialize weights as small numbers!
	Weights should neither be close to zero or one!
	'''
	
	# Initialize weights for hidden layer and input layer-
	# normal_numbers = tf.random.normal(shape = (3, 3), mean = 5.2, stddev=3.5, dtype=tf.float32)
	W1 = tf.random.uniform(shape = (input_layer, hidden_layer), minval = 0, maxval = 1, dtype=tf.float32) * tf.math.sqrt(2 / input_layer)

	# Initialize bias values for hidden layer-
	b1 = tf.random.uniform(shape = (1, hidden_layer), minval=0, maxval=1, dtype=tf.float32)

	# Initialize weights for output layer and hidden layer-
	W2 = tf.random.uniform(shape = (hidden_layer, output_layer), minval = 0, maxval = 1, dtype=tf.float32) * tf.math.sqrt(2 / (input_layer + hidden_layer))

	# Initialize bias values for output layer-
	# b2 = np.zeros((output_layer, 1))
	# OR-
	# b2 = np.random.randn(1, output_layer) * 0.01
	b2 = tf.random.uniform(shape = (1, output_layer), minval=0, maxval=1, dtype=tf.float32)

	# Return all weights and biases as a dictionary object-
	parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

	return parameters


def compute_cross_entropy_cost(A, Y):
	"""
	Function to compute cost using multi-class
	cross-entropy loss

	Arguments-
	A- Softmax output of output layer; shape- (m, c)
	Y-  groud truth labels vector; shape- (m, c)
	m - number of training examples
	c - number of classes

	Return-
	Multiclass cross-entropy cost
	"""

	# Number of training examples-
	m = Y.shape[0]

	# compute multiclass cross-entropy cost-
	logprobs = tf.math.multiply((tf.math.log(A) +1e-9), Y)
	cost = (-1 / m) * tf.math.reduce_sum(logprobs)

	# cost = -tf.math.reduce_mean(tf.matmul(tf.transpose(tf.math.log(A) + 1e-8), Y))

	# makes sure cost is the dimension we expect, E.g., turns [[51]] into 51-
	# cost = np.squeeze(cost)
	cost = tf.squeeze(cost)

	return cost


def forward_and_backward_propagation(X, parameters, Y, hidden_neurons):
	'''
	Function to compute forward propagation based on 'X', 'Y'
	and 'parameters' dictionary

	Input:
	1.) 'X' a tensor 
	2.) 'Y' a tensor
	3.) 'parameters' a Python dictionary containing weights & biases


	Returns:
	1.) gradients for weights and biases
	2.) cost

	Z1 is network input for hidden layer
	A1 is output of hidden layer (ReLU output)
	Z2 is network input for output layer
	A2 is output of output layer (softmax output)
	'''

	# Retreive initialized weights & biases-
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']


	# Forward propagation-

	# Network input for first hidden layer-
	Z1 = tf.matmul(tf.transpose(W1), tf.transpose(X)) + tf.transpose(b1)

	# Network output of first hidden layer using ReLU activation function-
	A1 = relu(Z1)

	# Network input for output layer-
	Z2 = tf.matmul(tf.transpose(A1), W2) + b2

	# Output of neural network using softmax activation function-
	A2 = softmax_stable(Z2)


	# Number of training examples-
	m = X.shape[0]


	# Backward propagation for finding partial derivatives of weights and biases
	# wrt to cost function-

	# Partial derivative of cost function wrt W2-
	dJ_dW2 = (1 / m) * tf.matmul(A1, (A2 - Y))

	# Sanity check-
	# dJ_dW2.shape == W2.shape
	# True

	# Partial derivative of cost function wrt b2-
	dJ_db2 = (1 / m) * tf.reshape(tf.math.reduce_sum((A2 - Y), axis = 0), shape = (1, 10))

	# dJ_db2.shape == b2.shape
	# True

	# Partial derivative of cost function wrt W1-
	dJ_dW1 = (1 / m) * tf.transpose(tf.matmul(tf.math.multiply(tf.transpose(tf.matmul((A2 - Y), tf.transpose(W2))), relu_derivative(A1)), X))                                                                                                  

	# Sanity check-
	# dJ_dW1.shape == W1.shape
	# True

	# Partial derivative of cost function wrt b1-
	dJ_db1 = (1 / m) * tf.reshape(tf.math.reduce_sum(tf.transpose(tf.math.multiply(tf.matmul(W2, tf.transpose((A2 - Y))), relu_derivative(A1))), axis = 0), shape = (1, hidden_neurons))

	cost = compute_cross_entropy_cost(A2, Y)

	gradient = {
		'dJ_dW1': dJ_dW1, 'dJ_dW2': dJ_dW2,
		'dJ_db1': dJ_db1, 'dJ_db2': dJ_db2
	}

	return gradient, cost


def optimization(wts_bias_parameters, X, Y, hidden_neurons, num_iterations, learning_rate, print_cost = False):
	'''
	Function to perform optimization to learn weight 'W' and bias 'b'
	by using gradient descent algorithm

	Returns the learnt 'W' and 'b' parameters AFTER training on training data
	'''

	# List variable to hold cost/loss-
	costs = []

	for i in range(num_iterations):
		# Compute gradients and cost using defined function-
		gradients, cost = forward_and_backward_propagation(X, wts_bias_parameters, Y, hidden_neurons)

		# Get partial derivates from 'gradients'
		dJ_db1 = gradients['dJ_db1']
		dJ_db2 = gradients['dJ_db2']
		dJ_dW2 = gradients['dJ_dW2']
		dJ_dW1 = gradients['dJ_dW1']

		W1 = wts_bias_parameters['W1']
		W2 = wts_bias_parameters['W2']
		b1 = wts_bias_parameters['b1']
		b2 = wts_bias_parameters['b2']

		# Update weights-
		W1 = W1 - (learning_rate * dJ_dW1)
		W2 = W2 - (learning_rate * dJ_dW2)

		# Update biases-
		b1 = b1 - (learning_rate * dJ_db1)
		b2 = b2 - (learning_rate * dJ_db2)

		# Update 'wts_bias_parameters' dict for next call-
		wts_bias_parameters = {'W1': W1, 'W2': W2,
				'b1': b1, 'b2': b2}

		# Store loss/cost AFTER every 100 iterations-
		if i % 100 == 0:
			costs.append(cost)

		# Print cost AFTER every 100 iterations-
		if print_cost and i % 100 == 0:
			print("\nLoss/Cost after {0} iterations = {1:.4f}\n".format(i, cost))


	# Update computed weights and biases-
	# The actual weights and biases AFTER training is done
	updated_params = {'W1': W1, 'W2': W2, 
				'b1': b1, 'b2': b2}

	# Update partial derivatives as a dictionary
	# Partial derivatives AFTER training is done-
	updated_gradients = {'dJ_dW2': dJ_dW2, 'dJ_dW1': dJ_dW1,
			'dJ_db1': dJ_db1, 'dJ_db2': dJ_db2}


	# Return everything-
	return updated_params, updated_gradients, costs


def predict(wts_bias_parameters, X):
	'''
	Function to make predictions using trained weights and biases

	Calculate A2 (output of neural network)
	'''

	# Number of examples-
	m = X.shape[0]


	# Network input for hidden layer-
	Z1 = tf.matmul(tf.transpose(W1), tf.transpose(X)) + tf.transpose(b1)

	# Network output of first hidden layer using ReLU activation function-
	A1 = relu(Z1)

	# Network input for output layer-
	Z2 = tf.matmul(tf.transpose(A1), W2) + b2

	# Output of neural network using softmax activation function-
	A2 = softmax_stable(Z2)




# Initialize parameters by specifying number of neurons in hidden layer-
n_hidden_neurons = 100
n_output_neurons = 10

initialized_params = initialize_parameters(X_train_reshaped.shape[1], n_hidden_neurons, n_output_neurons)
# 'initialized_params' is a dict


updated_params, updated_gradients, costs = optimization(
	wts_bias_parameters = initialized_params,
	X = X_train_reshaped, Y = y_train_ohe, hidden_neurons = n_hidden_neurons,
	num_iterations = 1000, learning_rate = 0.001, print_cost = True)

