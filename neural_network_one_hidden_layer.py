

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy


"""
Neural network with one hidden layer for binary classification
"""


ROWS = 64
COLS = 64
CHANNELS = 3

TRAIN_DIR = 'Train_data/'
TEST_DIR = 'Test_data/'

# Read full dataset-
train_images = [TRAIN_DIR + x for x in os.listdir(TRAIN_DIR)]
test_images = [TEST_DIR + x for x in os.listdir(TEST_DIR)]

# len(train_images), len(test_images)
# (6002, 1000)


def read_image(file_path):
    '''
    Function to read one image from 'file_path',
    resize it and return it for future use

    'file_path' is list
    '''
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)

    
    # resize our images-
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


# Example usage-
# image = read_image("Train_data/cat.0.jpg")

# image.shape
# (64, 64, 3)


def prepare_data(images):
    '''
    Function to prepare our data for further use. In this function, we
    will separate data, so if our image is with dog, we will give an index
    of 1 and if there will be a cat, we will give index of 0
    '''

    # Number of images is 'm'-
    m = len(images)

    # 'X' is training data-
    # X = np.zeros((m, ROWS, COLS, CHANNELS), dtype = np.unit8)
    X = np.zeros((m, ROWS, COLS, CHANNELS))

    # 'y' is labels/targets for training data in 'X'-
    y = np.zeros((1, m))

    for i, image_file in enumerate(images):
        X[i, :] = read_image(image_file)
        if 'dog' in image_file.lower():
            y[0, i] = 1
        elif 'cat' in image_file.lower():
            y[0, i] = 0

    return X, y


def initialize_parameters(input_layer, hidden_layer, output_layer):
	'''
	Function to initialize parameters for a neural network with-
	'input_layer' number of neurons in input layer
	'hidden_layer' number of neurons in hidden layer
	'output_layer' neuron in output layer [output layer has one neuron]

	Initialize weights as small numbers!
	'''
	# Initialize weights for hidden layer and input layer-
	W1 = np.random.randn(hidden_layer, input_layer) * 0.01

	# Initialize bias values for hidden layer-
	# b1 = np.zeros((hidden_layer, 1))
	# OR-
	# b1 = np.random.rand(hidden_layer, 1) * 0.01
	b1 = np.random.randn(1, hidden_layer) * 0.01

	# Initialize weights for output layer and hidden layer-
	W2 = np.random.randn(output_layer, hidden_layer) * 0.01

	# Initialize bias values for output layer-
	# b2 = np.zeros((output_layer, 1))
	# OR-
	# b2 = np.random.rand(output_layer, 1) * 0.01
	b2 = np.random.randn(1, output_layer) * 0.01

	# Return all weights and biases as a dictionary object-
	parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

	return parameters


def tanh_derivative(x):
	'''
	Function to calculate derivative of hyperbolic tangent
	for given parameter 'x'
	Used as activation function for hidden layer
	'''
	return (1 - np.power(np.tanh(x), 2))


def sigmoid(z):
	s = 1 / (1 + np.exp(-z))
	return s


def relu(x):
	'''
	Function to calculate ReLU for
	given 'x'
	'''
	return np.maximum(x, 0)


def relu_derivative(x):
	'''
	Function to calculate derivative
	of ReLU
	'''
	return np.where(x <= 0, 0, 1)


def visualize_tanh_and_derivative(x):
	'''
	Function to visualize tanh and it's derivative
	based on given input 'x'
	'''

	plt.plot(x, np.tanh(x), 'red', label = "hyperbolic tangent")
	plt.plot(x, tanh_derivative(x), 'g', label = "hyperbolic tangent derivative")
	plt.title("Hyperbolic tangent and it's derivative")
	plt.legend()
	plt.xlabel("x values")
	plt.ylabel("y values")
	plt.grid()
	plt.show()

	return None


# Generate x values-
# x = np.linspace(-10, 10, num = 100)

# x.shape
# (100,)

# visualize_tanh_and_derivative(x)


def forward_and_backward_propagation(X, parameters, Y):
	'''
	Function to compute forward propagation based on 'X', 'Y'
	and 'parameters' dictionary

	Input:
	1.) 'X' a numpy array 
	2.) 'Y' a numpy array
	3.) 'parameters' a Python dictionary containing weights & biases


	Returns:
	1.) gradients for weights and biases
	2.) cost

	Z1 is network input for hidden layer
	A1 is output of hidden layer (tanh output)
	Z2 is network input for output layer
	A2 is output of output layer (sigmoid output)
	'''

	# Retreive initialized weights & biases-
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']


	# Implement forward propagation to compute A2 probabilities-

	# Network input for hidden layer-
	Z1 = np.dot(W1, X.T) + b1.T		# (output_layer, m)

	# Using tanh as activation function for hidden layer-
	A1 = np.tanh(Z1)	# (output_layer, m)

	# Using ReLU as activation function for hidden layer-
	# A1 = np.maximum(Z1, 0)

	# Network input for output layer-
	Z2 = np.dot(W2, A1) + b2	# (n_o, m) OR (1, m)

	# Using sigmoid activation function for output layer
	A2 = sigmoid(Z2)	# (1, m) OR (n_o, m)

	# Number of training examples-
	m = X.shape[0]

	# Implement backward propagation for adjusting weights and biases-

	# Partial derivative of cost wrt W2-
	dJ_dW2 = (1 / m) * np.dot((A2 - y_train), A1.T)

	"""
	# Sanity check-
	if dJ_dW2.shape == W2.shape:
		print("\nW2.shape equals dJ_dW2.shape\n")
	else:
		print("\nW2.shape is NOT equal to dJ_dW2.shape! Recheck!\n")
	"""

	# Partial derivative of cost wrt W1 using tanh activation function-
	# dJ_dW1 = (1 / m) * np.dot(np.multiply(np.dot(W2.T, (A2 - Y)), (1 - np.square(A1))), X)

	# Partial derivative of cost wrt W1 using ReLU activation function-
	dJ_dW1 = (1 / m) * np.dot(np.multiply(np.dot(W2.T, (A2 - Y)), relu_derivative(A1)), X)

	# Sanity check-
	# dJ_dW1.shape == W1.shape
	# True

	# Partial detivative of cost wrt hidden layer bias (b1)- using tanh activation function-
	# dJ_db1 = (1 / m) * np.multiply(np.dot((A2 - Y), (1 - np.square(A1)).T), W2)

	# Partial derivative of cost wrt hidden layer bias (b1) using ReLU activation function-
	dJ_db1 = (1 / m) * np.multiply(np.dot((A2 - Y), relu_derivative(A1).T), W2)

	# Sanity check-
	# dJ_db1.shape == b1.shape
	# True

	# Partial derivative of cost wrt output layer bias (b2)-
	dJ_db2 = (1 / m) * np.sum(A2 - y_train)
	# Returns a scalar with shape ()

	cost = compute_cost(A2, Y)

	gradient = {
		'dJ_dW1': dJ_dW1, 'dJ_dW2': dJ_dW2,
		'dJ_db1': dJ_db1, 'dJ_db2': dJ_db2
	}

	return gradient, cost


# def compute_cost(A2, Y, parameters):
def compute_cost(A2, Y):
	'''
	Function to compute cost using binary cross-entropy
	loss

	Arguments-
	A2- Sigmoid output of output layer; shape- (1, m)
	Y-  groud truth labels vector; shape- (1, m)
	# parameters- Python dict containing W1, b1, W2, b2
	m - number of training examples

	Return-
	Binary cross-entropy cost
	'''

	# Number of training examples-
	m = Y.shape[1]

	# Compute binary cross-entropy cost-
	logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
	cost = (-1/m) * np.sum(logprobs)

	# np.multiply() multiplies arguments element-wise
	# np.log() natural logarithm, element-wise

	# makes sure cost is the dimension we expect, E.g., turns [[51]] into 51-
	cost = np.squeeze(cost)

	return cost


def optimization(wts_bias_parameters, X, Y, num_iterations, learning_rate, print_cost = False):
	'''
	Function to perform optimization to learn weight 'W' and bias 'b'
	by using gradient descent algorithm

	Returns the learnt 'W' and 'b' parameters AFTER training on training data
	'''

	# List variable to hold cost/loss-
	costs = []

	for i in range(num_iterations):
		# Compute gradients and cost using defined function-
		gradients, cost = forward_and_backward_propagation(X, wts_bias_parameters, Y)

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
		wts_bias_parameters = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

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

	Calculate A2 such that-
	If A2 <= 0.5, we predict a DOG
	Else, we predict a CAT (A2 > 0.5)

	Label for CAT is 1
	Label for DOG is 0
	'''

	# Number of examples-
	m = X.shape[0]

	y_pred = np.zeros((1, m))	# shape- (1, m)

	W1 = wts_bias_parameters['W1']
	W2 = wts_bias_parameters['W2']
	b1 = wts_bias_parameters['b1']
	b2 = wts_bias_parameters['b2']

	# Perform forward propagation to predict for given 'X'-
	Z1 = np.dot(W1, X_test_flattened.T) + b1.T		# (output_layer, m)
	A1 = np.tanh(Z1)	# (output_layer, m)
	Z2 = np.dot(W2, A1) + b2	# (n_o, m) OR (1, m)
	A2 = sigmoid(Z2)

	for i in range(A2.shape[1]):
		# Convert probabilities A2[0, i] to binary labels-
		if A2[0, i] > 0.5:
			y_pred[0, i] = 1		# 1 is for CAT
		else:
			y_pred[0, i] = 0		# 0 is for DOG

	return y_pred


# Now call our created function to read all test and train images we have in
# our folders-
X_train, y_train = prepare_data(train_images)
X_test, y_test = prepare_data(test_images)

print("\nX_train.shape = {0}, y_train.shape = {1}\n".format(X_train.shape, y_train.shape))
print("\nX_test.shape = {0}, y_test.shape = {1}\n".format(X_test.shape, y_test.shape))


# Reshape images to convert them into 1-Dimensional flattened arrays-
X_train_flattened = X_train.reshape(6002, 64 * 64 * 3)
X_test_flattened = X_test.reshape(1000, 64 * 64 * 3)

# Normalize training and testing sets-
X_train_flattened = X_train_flattened / 255.0
X_test_flattened = X_test_flattened / 255.0

print("\nDimensions of training and testing sets are:")
print("X_train_flattened = {0} & X_test_flattened = {1}\n".format(X_train_flattened.shape, X_test_flattened.shape))
# Dimensions of training and testing sets are:
# X_train_flattened = (6002, 12288) & X_test_flattened = (1000, 12288)

print("\nDimensions of training and testing labels are: ")
print("y_train = {0} & y_test = {1}\n".format(y_train.shape, y_test.shape))
# Dimensions of training and testing labels are: 
# y_train = (1, 6002) & y_test = (1, 1000)


# Initialize parameters by specifying number of neurons in each layers-

n_hidden_neurons = 10
n_input_neurons = X_train.flattened.shape[1]

initialized_params = initialize_parameters(n_input_neurons, n_hidden_neurons, 1)
# 'initialized_params' is a dict

# Train the model for 1000 epochs-
updated_params, updated_gradients, costs = optimization(wts_bias_parameters = initialized_params,
	X = X_train_flattened, Y = y_train, num_iterations = 1000,
	learning_rate = 0.001, print_cost = True)


# Make predictions on test set using trained model-
y_predictions = predict(updated_params, X_test_flattened)

# Make predictions on training set using trained model-
y_predictions_train = predict(updated_params, X_train_flattened)

# Calculate accuracy of TRAINED neural network on test and train sets-
accuracy_test = np.mean(np.abs(y_predictions - y_test) * 100)
accuracy_train = np.mean(np.abs(y_predictions_train - y_train) * 100)

print("\nAccuracy of trained neural network:")
print("Test set = {0:.4f} and Train set = {1:.4f}\n".format(accuracy_test, accuracy_train))
# Accuracy of trained neural network:
# Test set = 49.5000 and Train set = 50.5498



"""
# OPTIONAL: Save trained 'updated_params', 'updated_gradients' and 'costs' as pickled
# objects for later use-
import pickle

# Save trained weights and biases-
with open("/path_to_file/trained_wts_bias_4_hidden_neurons.pkl", "wb") as f:
	pickle.dump(updated_params, f)


# Save trained gradients-
with open("/path_to_file/trained_gradients_4_hidden_neurons.pkl", "wb") as f:
	pickle.dump(updated_gradients, f)


# Save costs-
with open("/path_to_file/costs_4_hidden_neurons.pkl", "wb") as f:
	pickle.dump(costs, f)

"""
