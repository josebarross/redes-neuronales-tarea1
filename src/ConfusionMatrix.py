import numpy as np
import pandas as pd


# A class that represents the confusion matrix
class ConfusionMatrix:
    # Receives a list of predicted values and actual labels and created confusion matrix
    def __init__(self, predicted, labels):
        name_of_labels = set(labels)
        self.name_to_index = {}
        for i, name in enumerate(name_of_labels):
            self.name_to_index[name] = i
        number_of_labels = len(self.name_to_index)
        self.matrix = np.zeros((number_of_labels, number_of_labels))
        for prediction, label in zip(predicted, labels):
            self.matrix[self.name_to_index[prediction]][self.name_to_index[label]] += 1

    # Gets the precision of a label
    def precision_of_label(self, label):
        i = self.name_to_index[label]
        summ = sum(self.matrix[i]) or 0.0
        return self.matrix[i][i] / summ

    # Gets the recall of a label
    def recall_of_label(self, label):
        i = self.name_to_index[label]
        summ = sum(self.matrix[:, i]) or 0.0
        return self.matrix[i][i] / summ

    # Gets the total precision of the confusion maatrix (Sum of diagonal divided by total amount of examples
    def total_precision(self):
        total = sum([sum(row) for row in self.matrix])
        return sum([self.matrix[i][i] for i in range(len(self.matrix))]) / total

    # Prints the confusion matrix, and the recall and precision for each category.
    def matrix_summary(self):
        for label in self.name_to_index:
            print(
                "Category {} has precision={} and recall={}".format(
                    label, self.precision_of_label(label), self.recall_of_label(label)
                )
            )
        print("Confusion Matrix details: ")
        # This code is to print it as a pandas dataframe so it is prettier
        printable = {}
        printable["x"] = [x for x in self.name_to_index]
        for i, x in enumerate(self.name_to_index):
            printable[x] = self.matrix[:, i]
        print(pd.DataFrame(printable))
