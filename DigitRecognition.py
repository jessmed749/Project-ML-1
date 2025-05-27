import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the data for CNN input
x_train = x_train.reshape((-1, 28, 28, 1))  # Reshape to (num_samples, height, width, channels)
x_test = x_test.reshape((-1, 28, 28, 1))  # Reshape to (num_samples, height, width, channels)

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

