# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import math
import random

import numpy as np
from Utils import softmax, sigmoid, categorical_cross_entropy, loss_derivative_wrt_activations, derivative_sigmoid, \
    derivative_relu, \
    one_hot, derivative_softmax, loss_derivative_wrt_z


class Perceptron:
    def __init__(self):
        pass

    # Activation function
    def step_func(self, x):
        if x >= 0:
            return 1
        return 0

    # Helper to calculate predicted value for given data, weights and bias
    def predict(self, X, W, b):
        res = np.dot(X, W) + b
        return self.step_func(res)

    # Training over one epoch
    def train_epoch(self, X, Y, W, b, learning_rate, indices):
        ## Total error
        error = 0

        for i in indices:
            # Calculate prediction and loss
            prediction = self.predict(X[i], W, b)
            loss = Y[i] - prediction

            # Update weights and bias
            for j in range(X.shape[1]):
                W[j] += X[i][j] * learning_rate * loss
            b += learning_rate * loss

            # Update total error
            error += abs(loss)

        return W, b, error / indices.shape[0]

    # Training over multiple epochs
    def train(self, X, Y, W, b, learning_rate, epochs):
        # Errors over epochs
        errors = []

        for e in range(epochs):
            # Shuffle data
            random_indices = np.arange(X.shape[0])
            np.random.shuffle(random_indices)

            # Train one epoch and store error
            W, b, error = self.train_epoch(X, Y, W, b, learning_rate, random_indices)
            errors.append(error)

        return W, errors


class Layer:

    def __init__(self, n, prev_layer=None, weights=None, biases=None, output=False):
        self.n = n
        self.prev_layer = prev_layer  # We store this to link the network
        self.weights = weights
        self.biases = biases
        # This is used to switch the activation function in feedforward.
        # It is set to true in case it is the output layer
        self.output = output
        # self.error = None

    def set_weights(self, weights):
        self.weights = weights

    def init_weights(self, deterministic=False):
        """
        Initialises the weight that is normally distributed with mean = 0 and sd = 1 - (n_prev_layer)

        :return: Returns the weights that is normally distributed according to the Xavier Initialization

        """
        if deterministic:
            seed = 42
            np.random.seed(seed)

        n_prev_layer = self.prev_layer.n
        shape = (self.n, self.prev_layer.n)
        self.weights = np.random.normal(0, 1 / n_prev_layer, shape)

    def set_biases(self, biases):
        self.biases = biases

    def feed_forward_layer(self, x, activation_function):
        """
        Returns the output. output (a) = activation_function(W * X + b)
        Activation function used in case of the output layer is softmax
        In case of the hidden layers, it is sigmoid (or can also be relu)

        :param activation_function: the activation function to be used. Can be sigmoid, relu or softmax
        :param x: the input vector or the vector containing activation from the previous layer
        :return: the output vector
        """
        z = (self.weights @ x) + np.reshape(self.biases, (-1, 1))
        a = activation_function(z)
        return np.reshape(a, (-1, 1)), np.reshape(z, (-1, 1))


class ANN:
    """
    It has a List of Layer object
    """

    def __init__(self, layers, hidden_layer_activation, derivative_activation):
        self.layers = layers
        self.hidden_layer_activation = hidden_layer_activation  # The activation function to be used in the hidden layers
        self.derivative_activation = derivative_activation

    def feedforward(self, x):
        """
        This goes layer by layer using the Layer.feed_forward_layer. It takes in the input, computes the activation
        for the next layer, then takes that activation as the input and computes it for next layer and so on...

        :param x: the input vector
        :return: the output vector
        """
        x = np.reshape(x, (-1, 1))
        a = x
        activations = [x]
        zs = []
        for layer in self.layers[1:]:
            if layer.output:
                a, z = layer.feed_forward_layer(a, softmax)
            else:
                a, z = layer.feed_forward_layer(a, self.hidden_layer_activation)
            # I save all the activations and zs because it is required in back propagation
            activations.append(np.reshape(a, (-1, 1)))
            zs.append(np.reshape(z, (-1, 1)))
        return a, activations, zs

    def back_propagate(self, x, y, regularisation_constant, n):
        """
        This function computes the cost gradient wrt w and b for one single data point
        The formula I used for this can be found in this book:
        http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm

        :param n: Size of the training set
        :param regularisation_constant: (λ) the regularisation constant
        :param x: Input vector
        :param y: Output vector
        :return: returns the gradients
        """
        activations, zs = self.feedforward(x)[1:]
        L = len(self.layers)  # number of layers in our networks
        n_classes = self.layers[-1].n  # no of output classes

        # d_bs stores derivative of Cost wrt to the biases for each layer. Initially, it is all set to 0.
        # d_ws stores derivative of Cost wrt to the weights for each layer. Initially, all 0.
        d_bs = [np.zeros(self.layers[l].biases.shape) for l in range(1, L)]
        d_ws = [np.zeros(self.layers[l].weights.shape) for l in range(1, L)]

        # Back propagate for the output layer
        d_cost_wrt_z = loss_derivative_wrt_z(activations[-1], one_hot(y, n_classes))

        # This reshaping helps make the computation easier. I change it from (n,) to (n,1) because .
        d_cost_wrt_z = np.reshape(d_cost_wrt_z, (-1, 1))

        # I use delta throughout this function, and it is basically the error (dC/dz) for each neuron in a layer.
        # I start by computing it for the output layer.
        # This intermediate value makes it easier to compute dC/dW and dC/db
        delta = d_cost_wrt_z

        # delta = delta.transpose()  # I get a row. I need  a column vector.

        # For the outermost layer, I set the d_b to be delta. Delta is just dC/dz (check formula from the book)
        d_bs[-1] = delta
        # For the output layer, I set the d_w according to the formula from the book.
        # dC/dW = (activation from prev layer) * (delta for the current layer)
        # (regularisation_constant/n) * self.layers[-1].weights is the regularisation factor
        d_ws[-1] = delta @ activations[-2].transpose() + (regularisation_constant / n) * self.layers[-1].weights
        for l in range(2, L):
            front_layer = self.layers[-l + 1]
            layer = self.layers[-l]
            z = zs[-l]
            # This again uses the formula. Delta (δ^l) for the layer l is (W^(l+1))(δ^(l+1)) * (σ'(z^l))
            # * is the hadamard product or simply put, element wise multiplication
            delta = (front_layer.weights.transpose() @ delta) * self.derivative_activation(z)

            # I set the changes according to the formula once again
            d_bs[-l] = delta
            d_ws[-l] = delta @ activations[-l - 1].transpose() + (regularisation_constant / n) * self.layers[-l].weights
        return d_bs, d_ws

    def train_network(self, train_x, train_y, val_x, val_y, epochs=100, batch_size=8, learning_rate=0.1,
                      regularisation_factor=0.0):
        L = len(self.layers)
        n_classes = self.layers[-1].n

        # Store initial training/validation results
        train_loss, train_acc, train_confusion_matrix, train_predictions = self.evaluate(train_x, train_y, regularisation_factor)
        val_loss, val_acc, val_confusion_matrix, val_predictions = self.evaluate(val_x, val_y, regularisation_factor)
        train_losses = [train_loss]
        val_losses = [val_loss]
        train_accuracies = [train_acc]
        val_accuracies = [val_acc]

        # Stores the gradient of cost wrt b and w Initially set to 0. I keep adding the gradients I compute using
        # backprop for each data point to this At the end, while updating the weights and biases, I take the average
        # of the gradients over all the data points in that batch
        d_bs = [np.zeros((self.layers[l].biases.shape[0], 1)) for l in range(1, L)]
        d_ws = [np.zeros(self.layers[l].weights.shape) for l in range(1, L)]

        for epoch in range(1, epochs + 1):
            print("Starting epoch ", epoch, "...")

            data = list(zip(train_x, train_y))
            random.shuffle(data)

            for batch_index in range(math.ceil(len(train_x) / float(batch_size))):
                batch = data[batch_index * batch_size: min((batch_index + 1) * batch_size, len(train_x))]

                # I reset the gradients to 0 because I have to compute it once again for the new batch
                d_bs = [np.zeros((self.layers[l].biases.shape[0], 1)) for l in range(1, L)]
                d_ws = [np.zeros(self.layers[l].weights.shape) for l in range(1, L)]

                for inputX, outputY in batch:
                    delta_b, delta_w = self.back_propagate(inputX, outputY, regularisation_factor, len(train_x))

                    # Here, I add the gradient of C wrt b and W for the current datapoint to the already existing sum
                    # The sum is of gradients of all the datapoints we have seen so far in the batch
                    d_bs = [np.reshape(nb, (-1, 1)) + np.reshape(dnb, (-1, 1)) for nb, dnb in zip(d_bs, delta_b)]
                    d_ws = [nw + dnw for nw, dnw in zip(d_ws, delta_w)]

                # Update weights and biases for each layer
                for l in range(1, L):
                    layer_weight = self.layers[-l].weights
                    layer_bias = np.reshape(self.layers[-l].biases, (-1, 1))
                    self.layers[-l].weights = layer_weight - ((d_ws[-l] / len(batch)) * learning_rate)
                    self.layers[-l].biases = layer_bias - ((d_bs[-l] / len(batch)) * learning_rate)

            # Store training/validation results after the epoch:
            train_loss, train_acc, train_confusion_matrix, train_predictions = self.evaluate(train_x, train_y, regularisation_factor)
            val_loss, val_acc, val_confusion_matrix, val_predictions = self.evaluate(val_x, val_y, regularisation_factor)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

        return train_losses, val_losses, train_accuracies, val_accuracies, train_confusion_matrix, val_confusion_matrix, \
            train_predictions, val_predictions

    def sum_of_square_of_weights(self):
        sum_v = 0
        L = len(self.layers)
        for i in range(1, L):
            sum_v += np.square(self.layers[i].weights).sum()
        return sum_v

    def evaluate(self, test_x, test_y, regularisation_factor):
        n_classes = self.layers[-1].n
        test_results = []
        loss = 0
        confusion_matrix = np.zeros((n_classes, n_classes))

        for (x, y) in zip(test_x, test_y):
            ires = self.feedforward(x)[0]
            loss += categorical_cross_entropy(ires, one_hot(y, n_classes))
            y_pred = np.argmax(ires)
            confusion_matrix[y_pred][y - 1] += 1
            # Adding 1 to y_pred because the argmax() returns number starting from 0
            test_results.append((y_pred + 1, y))
        avg_loss = loss/len(test_x)
        regularised_loss = avg_loss + (regularisation_factor / (2 * len(test_y))) * self.sum_of_square_of_weights()
        res = [int(x == y) for (x, y) in test_results]
        predictions = [x for (x, y) in test_results]
        return regularised_loss, sum(res) / len(test_x), confusion_matrix, predictions

    def sum_of_weights(self):
        sum_v = 0
        L = len(self.layers)
        for l in range(1, L):
            sum_v += self.layers[l].weights.sum()
        return sum_v