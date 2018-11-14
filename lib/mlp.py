import math
import sys

import numpy as np


class MLP:
    """"
    Multi-layer perceptron, from scratch.

    Weights are learned using back-propagation with mini-batches gradient descent.
    Activation function is the sigmoid function. Loss function is the mean squared error.

    Arguments:
        layers (list of int): the number of neurons in each layer of the network
          including the mandatory input and output layers.

    The meaning of the various 1-letter variables is as follow:

     * L: the number of layers in the network, including the mandatory input and output layers

     * l: the index of a layer in the network

     * N: the size of the dataset

     * x: the dataset
          numpy array of size D x N, with D the input data dimension. Note that layers[0] = D, necessarily.

     * y: the labels
          numpy array of size d x N, with d the labels dimension. Note that layers[-1] = d, necessarily.

     * w: the weights of the network
          list of size L of numpy arrays of size layers[l] x layers[l+1]

     * z: the output of the neurons in the network before activation function
          list of size L of numpy arrays of size layer[l] x batch_size

     * b: the bias for each neuron in the network
          list of size L - 1 of numpy arrays of size layer[l] x batch_size

     * a: the output of the neurons in the network after activation function
          same shape as z
     """

    def __init__(self, layers):
        if len(layers) < 2:
            raise ValueError('at least an input and output layer must be specified')

        self.layers = layers
        self.L = len(self.layers)

        self.init_weights()

        self.epoch = 0
        self.errors = []

    def init_weights(self):
        np.random.seed(2)
        self.w = []
        self.b = []
        for previous_l, l in zip(self.layers, self.layers[1:]):
            factor = 1. / math.sqrt(l)
            self.w.append(factor * np.random.randn(l, previous_l))
            self.b.append(factor * np.random.randn(l, 1))
        self.w.append(np.identity(self.layers[-1]))

    def reset(self):
        self.epoch = 0
        self.errors = []
        self.init_weights()

    def fit(self, x, y, epochs, batch_size=1, eta=0.1, verbose=False):
        """
        Train the neural network.

        Arguments:
            x (numpy array): the dataset
            y (numpy array): the labels
            epochs (int): the number of epochs
            batch_size (int, optional): the size of the mini-batches
            eta (float, optional): the learning rate
            verbose (bool, optional): whether to print error at each epoch while training

        Returns:
            Float error on the training set.

        Notes:
            This function can be called multiple times. Learning is resumed every time.
            To reset the network, use ``self.reset()``.

            Errors for each epoch are accumulated in ``self.errors``.
        """

        if self.epoch == 0:
            self.errors.append(self.evaluate(x, y))
            self.epoch += 1

        # Run epochs
        max_epoch = self.epoch + epochs
        while self.epoch < max_epoch:

            error = self.fit_epoch(x, y, batch_size, eta)
            self.errors.append(error)

            if verbose:
                print('epoch %i: mse = %.4f' % (self.epoch, error))

            self.epoch += 1

        return error

    def fit_epoch(self, x, y, batch_size, eta):
        batches = self.get_batches(x, y, batch_size)
        for x_batch, y_batch in batches:
            self.fit_batch(x_batch, y_batch, eta)
        error = self.evaluate(x, y)
        return error

    def fit_batch(self, x, y, eta):
        z, a = self.propagate_forward(x)
        delta = self.propagate_backward(a, y, z)
        self.w, self.b = self.update_weights(a, delta, eta)

    def get_batches(self, x, y, batch_size):
        N = x.shape[1]
        indices = range(N)
        shuffled_indices = np.random.permutation(indices)

        for i in range(0, N, batch_size):
            batch_indices = shuffled_indices[i:i + batch_size]
            yield (x[:, batch_indices], y[:, batch_indices])

    def predict(self, x):
        _, a = self.propagate_forward(x)
        return a[-1]

    def evaluate(self, x, y):
        preds = self.predict(x)
        return self.mse(preds, y)

    def propagate_forward(self, x):
        a = [x]
        z = [x]
        for l in range(self.L - 1):
            z_l = np.dot(self.w[l], a[l]) + self.b[l]
            z.append(z_l)
            a.append(self.sigmoid(z_l))
        return z, a

    def propagate_backward(self, a, y, z):
        delta_l = delta_L = self.d_mse(a[-1], y) * self.d_sigmoid(z[-1])
        delta = [delta_L]
        for l in list(range(self.L - 1))[::-1]:
            delta_l = np.dot(self.w[l].T, delta_l) * self.d_sigmoid(z[l])
            delta = [delta_l] + delta
        return delta

    def update_weights(self, a, delta, eta):
        updated_w = []
        updated_b = []
        N = a[0].shape[1]
        for l in range(self.L - 1):
            updated_w.append(self.w[l] - eta / N * np.dot(delta[l + 1], a[l].T))
            updated_b.append(self.b[l] - eta / N * np.dot(delta[l + 1], np.ones((N, 1))))
        updated_w.append(np.identity(self.layers[-1]))
        return updated_w, updated_b

    @staticmethod
    def sigmoid(t):
        def scalar_sigmoid(t):
            return 1. / (1 + math.exp(-t))

        return np.vectorize(scalar_sigmoid)(t)

    @staticmethod
    def d_sigmoid(t):
        return MLP.sigmoid(t) * (1 - MLP.sigmoid(t))

    @staticmethod
    def mse(a, y):
        return (0.5 * ((a - y) ** 2).sum(axis=0)).mean()

    @staticmethod
    def d_mse(a, y):
        return a - y
