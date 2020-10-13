import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Evaluate the error (i.e., cost) between the prediction made in A2 and the provided labels Y
# We use the Mean Square Error cost function
def mean_square_error(A2, Y):
    cost = np.sum((0.5 * (A2 - Y) ** 2).mean(axis=1)) / A2.shape[1]
    return cost


# Normalizes data between the range given as nrange.
def normalize_data(data, nrange=[0, 1]):
    norm_data = []
    for i in range(data.shape[1]):
        norm_data.append(normalize_column(data[:, i], nrange))
    return np.array(norm_data).T


# Normalizes by column
def normalize_column(data, nrange):
    max_value = max(data)
    min_value = min(data)
    norm = (
        lambda x: (x - min_value) * (nrange[1] - nrange[0]) / (max_value - min_value)
        + nrange[0]
    )
    return [norm(x) for x in data]


# Implements one hot enconding.
# labels is the row of labels
# returns label hashmap with encode of each label so you can decode it later if you want.
def one_hot_encoding(labels):
    label_names = set(labels)
    n_of_labels = len(label_names)
    label_encode = {}
    for i, name in enumerate(label_names):
        encode = [0] * n_of_labels
        encode[i] = 1
        label_encode[name] = encode

    return label_encode, np.array([label_encode.get(x) for x in labels])


# Does a train test split. This split is random and also it leaves the
# same amount of examples for each category on the test set.
# Return train_data, test_data, train_labels, test_labels
def split_and_shuffle_train_test(data, labels, percentaje_train=0.8):
    test_size = int(len(data) * (1 - percentaje_train))
    set_labels = set(labels)
    # joins train and test data on same table, expand dims is used to be able to concatenate.
    # this is done to be able to select random without altering the desired label.
    mixed = np.concatenate((data, np.expand_dims(labels, axis=1)), axis=1)
    # Shufles mixed data and labels
    np.random.shuffle(mixed)
    test_data = []
    train_data = []
    # Adds first of each label to test data and others to train data.
    for label in set_labels:
        i = 0
        for row in mixed:
            if i <= test_size / len(set_labels) and row[-1] == label:
                i += 1
                test_data.append(row)
            elif row[-1] == label:
                train_data.append(row)
    test_data = np.array(test_data)
    train_data = np.array(train_data)
    return (
        train_data[:, :-1],
        test_data[:, :-1],
        train_data[:, -1],
        test_data[:, -1],
    )
