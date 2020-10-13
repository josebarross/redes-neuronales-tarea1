# Install Instructions
To run first install python. This was run with Python 3.8.
 - Create and activate virtual enviroment
 - Install dependencies (only used numpy, matplotlib and pandas)
 `pip install -r requirements.txt`
 - Run:
 `python src/__init__.py`
 
 # Code structure
 - init.py has the script that uses all functions on the IRIS dataset
 - FFNeuralNetwork.py has class that implements network
 - utils.py has various functions including sigmoid, one_hot_encoding, normalize_data, train_test_split
 - ConfusionMatrix.py has class that implements a confusion matrix
