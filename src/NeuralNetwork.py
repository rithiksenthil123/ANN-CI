# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np

class Perceptron:
    def __init__(self):
        pass

    ## Activation function
    def step_func(self, x):
        if x >= 0:
            return 1
        return 0

    ## Helper to calculate predicted value for given data, weights and bias
    def predict(self, X, W, b):
        res = np.dot(X,W)+b
        return self.step_func(res)

    ## Training over one epoch
    def train_epoch(self, X, Y, W, b, learning_rate, indeces):
        ## Total error
        error = 0

        for i in indeces:
            ## Calculate prediction and loss
            prediction = self.predict(X[i], W, b)
            loss = Y[i] - prediction

            ## Update weights and bias
            for j in range(X.shape[1]):
                W[j] += X[i][j] * learning_rate * loss
            b += learning_rate * loss

            ## Update total error
            error += abs(loss)

        return W, b, error

    ## Training over multiple epochs
    def train(self, X, Y, W, b, learning_rate, epochs):
        ## Errors over epochs
        errors = []

        for e in range(epochs):
            ## Shuffle data
            random_indices = np.arange(X.shape[0])
            np.random.shuffle(random_indices)

            ## Train one epoch and store error
            W, b, error = self.train_epoch(X, Y, W, b, learning_rate, random_indices)
            errors.append(error)

        return W, errors

class ANN:
    def __init__(self):
        pass