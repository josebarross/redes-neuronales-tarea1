import numpy as np
import matplotlib.pyplot as plt
from FFNeuralNetwork import TwoLayerNetwork
from ConfusionMatrix import ConfusionMatrix
from sklearn import datasets
from utils import one_hot_encoding, normalize_data, split_and_shuffle_train_test


def iris_dataset_classification():
    # Set the seed to make result reproducible
    np.random.seed(50)
    iris = datasets.load_iris()
    train_data, test_data, train_labels, test_labels = split_and_shuffle_train_test(
        iris.data, iris.target
    )
    encoding, train_labels = one_hot_encoding(train_labels)
    train_labels = train_labels.T
    train_data = normalize_data(train_data).T
    hidden_layer_neurons = range(200, 600, 30)
    total_precision = []
    for neuron in hidden_layer_neurons:
        network = TwoLayerNetwork(4, neuron, 3)
        network.train(train_data, train_labels, 20000, 0.0005)
        network.plot_training_cost()
        predictions = network.predict(test_data.T)
        cm = ConfusionMatrix(predictions, test_labels)
        cm.matrix_summary()
        total_precision.append(cm.total_precision())
    plot_precision_vs_number_neurons(hidden_layer_neurons, total_precision)


def plot_precision_vs_number_neurons(
    number_of_neurons, total_precision, index_figure=1
):
    plt.figure(index_figure)
    plt.title("Precision vs amount of neurons in hidden layer")
    plt.ylabel("Precision")
    plt.xlabel("Number of neurons")
    plt.plot(number_of_neurons, total_precision)
    plt.savefig("precision_vs_neurons.png")


def xor_classification():
    # Set the seed to make result reproducible
    np.random.seed(42)

    # The 4 training examples by columns
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

    # The outputs of the XOR for every example in X
    Y = np.array([[0, 1, 1, 0]])

    # Set the hyperparameters
    n_x = 2  # No. of neurons in first layer
    n_h = 8  # No. of neurons in hidden layer
    n_y = 1  # No. of neurons in output layer

    # The number of times the model has to learn the dataset
    number_of_iterations = 10000
    learning_rate = 0.01

    # define a model
    network = TwoLayerNetwork(n_x, n_h, n_y)
    network.train(X, Y, number_of_iterations, learning_rate)

    # Test 2X1 vector to calculate the XOR of its elements.
    # You can try any of those: (0, 0), (0, 1), (1, 0), (1, 1)
    X_test = np.array([[0], [1]])
    network.predict(X_test)


iris_dataset_classification()
