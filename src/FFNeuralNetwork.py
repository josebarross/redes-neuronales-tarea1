import numpy as np
from utils import mean_square_error, sigmoid
import matplotlib.pyplot as plt

# Class that implements the One hidden layer Feed Forward network
class TwoLayerNetwork:
    # n_x: number of inputs (this value impacts how X is shaped)
    # n_h: number of neurons in the hidden layer
    # n_y: number of neurons in the output layer (this value impacts how Y is shaped)
    def __init__(self, n_x, n_h, n_y):
        self.W1 = np.random.randn(n_h, n_x)
        self.b1 = np.zeros((n_h, 1))
        self.W2 = np.random.randn(n_y, n_h)
        self.b2 = np.zeros((n_y, 1))
        self.data = {}
        self.grads = {}
        self.cost_evolution = []

    # Evaluate the neural network
    def forward_prop(self, X):
        # Z value for Layer 1
        Z1 = np.dot(self.W1, X) + self.b1
        # Activation value for Layer 1
        A1 = np.tanh(Z1)
        # Z value for Layer 2
        Z2 = np.dot(self.W2, A1) + self.b2
        # Activation value for Layer 2
        A2 = sigmoid(Z2)

        self.data = {"A1": A1, "A2": A2}
        return A2

    # Apply the backpropagation
    def backward_prop(self, X, Y):
        A1 = self.data["A1"]
        A2 = self.data["A2"]
        m = X.shape[1]

        # Compute the difference between the predicted value and the real values
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        # Because d/dx tanh(x) = 1 - tanh^2(x)
        dZ1 = np.multiply(np.dot(self.W2.T, dZ2), 1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # Third phase of the learning algorithm: update the weights and bias
    def update_parameters(self, learning_rate):
        dW1 = self.grads["dW1"]
        db1 = self.grads["db1"]
        dW2 = self.grads["dW2"]
        db2 = self.grads["db2"]

        self.W1 = self.W1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1
        self.W2 = self.W2 - learning_rate * dW2
        self.b2 = self.b2 - learning_rate * db2

    # X: is the set of training inputs
    # Y: is the set of training outputs
    def train(self, X, Y, num_of_iters, learning_rate):
        for i in range(0, num_of_iters + 1):
            result = self.forward_prop(X)
            cost = mean_square_error(result, Y)
            self.backward_prop(X, Y)
            self.update_parameters(learning_rate)
            if i % 100 == 0:
                self.cost_evolution.append(cost)

    # Make a prediction
    # X: represents the inputs
    # the result is the prediction
    def predict(self, X):
        a2 = self.forward_prop(X)
        return np.argmax(a2, axis=0)

    # plots and saves the evolution of the training cost
    def plot_training_cost(self, index_figure=0):
        plt.figure(index_figure)
        plt.title("Training cost evolution per iteration")
        plt.ylabel("Cost")
        plt.xlabel("Number of iteration")
        plt.plot(range(0, 100 * len(self.cost_evolution), 100), self.cost_evolution)
        plt.savefig("cost_evolution.png")
