import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt


def three_way_split_data(features_file_name, targets_file_name, training_percentage, validation_percentage,
                         test_percentage, seed=1):
    """
    It takes the input data and splits it into training, validation and test data

    :param features_file_name: file name of the features
    :param targets_file_name: file name of the targets
    :return: 6-tuple with X_train, y_train, X_val, y_val, X_test, y_test
    """
    features = pd.read_csv("../data/features.txt", header=None).astype(float)
    targets = pd.read_csv("../data/targets.txt", header=None).astype(int)

    featuresDF = pd.DataFrame(features)
    targetsDF = pd.DataFrame(targets)

    data = pd.concat([featuresDF, targetsDF], axis=1)
    data = data.set_axis(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'target'], axis='columns')
    print(data)

    # Randomize and split the data in the different sets
    train_set = data.sample(frac=training_percentage, random_state=seed, axis=0)
    intermediate = data.drop(index=train_set.index)
    validation_set = intermediate.sample(frac=validation_percentage / (1 - training_percentage), random_state=seed,
                                         axis=0)
    test_set = intermediate.drop(index=validation_set.index).sample(frac=1.0, axis=0)

    X_train = train_set[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
    y_train = train_set[['target']]

    X_val = validation_set[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
    y_val = validation_set[['target']]

    X_test = test_set[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
    y_test = test_set[['target']]

    return X_train.to_numpy(), y_train.to_numpy(), X_val, y_val, X_test, y_test


def softmax(x):
    """
        Computes the softmax

        :param x: the list of input values
        :return: returns the list containing softmax for each of the x's
        """
    e_x = np.exp(
        x - np.max(x))  # The - np.max(x) is to avoid overflows. It does not change the final value of the softmax
    return e_x / e_x.sum()


def sigmoid(x):
    """
    The sigmoid activation of X. Gives the probability.
    :param x: the input vector
    :return: the output vector after applying the sigmoid activation to each x's in the inout vector
    """
    denominator = 1 + np.exp(-x)
    return 1 / denominator


def relu(x):
    """
    Rectified linear unit activation function. It returns x if x > 0. Otherwise, 0
    :param x: Input vector
    :return: Output vector that applied ReLU to each input value
    """
    return np.maximum(0, x)


def categorical_cross_entropy(y_predicted, y_expected):
    """

    :param sum_of_square_of_weights: The sum of (weights)^2 over all the weights in the network
    :param n: size of training set
    :param reg_factor: (λ) the regularisation factor
    :param y_predicted: The predicted output vector
    :param y_expected: The expected output vector
    :return: The computed loss
    """
    #
    # print("Shape of predicted: ", y_predicted.shape)
    # print("Shape of expected: ", y_expected.shape)
    loss = np.dot(y_expected.transpose(), np.log(y_predicted))
    # loss = y_expected.transpose() @ np.log(y_predicted)

    return -loss.sum()

def one_hot(class_value, number_of_classes):
    res = np.zeros(number_of_classes)
    res[class_value - 1] = 1
    return np.reshape(res, (-1, 1))


def loss_derivative_wrt_activations(y_exp, activations):
    """
    Derivative of the loss wrt z
    Some complicated math but yes it ends up being this simple.
    Idea: https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1

    :param y_exp: one hot vector of expected output
    :param activations: the activations of the output layer
    :return: The computed derivative
    """
    return np.divide(y_exp, activations)


def loss_derivative_wrt_z(y_pred, y_exp):
    return y_pred - y_exp


def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def derivative_relu(z):
    return np.where(z > 0, 1, 0)


def derivative_softmax(z):
    """
    Source: the idea for this was derived from https://e2eml.school/softmax.html
    :param z: the vector for which you want to evaluate the derivative of softmax for
    :return: the derivative
    """
    softmax_v = softmax(z)
    softmax_v = np.reshape(softmax_v, (1, -1))  # reshaping makes the computation easy
    return softmax_v * np.identity(softmax_v.size) - softmax_v.transpose() @ softmax_v


def plot_lines(title, x_title, y_title, first_legend, second_legend, first_data, second_data, max_x):
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.plot(np.arange(max_x + 1), first_data, label=first_legend)
    plt.plot(np.arange(max_x + 1), second_data, label=second_legend)
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()


def plot_points(title, x_title, y_title, first_legend, second_legend, first_data, second_data, x_data):
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.xlim([0, 40])
    plt.xticks(x_data, x_data)
    plt.plot(x_data, first_data, "o", label=first_legend)
    plt.plot(x_data, second_data, "x", label=second_legend)
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()
