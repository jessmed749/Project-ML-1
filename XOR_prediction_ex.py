#create a Neural Network (NN) that can properly predict
# values from the XOR function.

import numpy as np

# Hyperparameters
epochs = 60000  # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 3, 1
LR = 0.1  # Learning rate

# XOR input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights with random values
w_hidden = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
w_output = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return x * (1 - x)

# Training the neural network
for epoch in range(epochs):
    # Forward pass
    act_hidden = sigmoid(np.dot(X, w_hidden))
    output = sigmoid(np.dot(act_hidden, w_output))
    
    # Calculate error
    error = y - output
    
    if epoch % 5000 == 0:
        print(f'Epoch {epoch}: error sum {np.sum(error)}')
    
    # Backward pass
    dZ_output = error * LR * sigmoid_prime(output)
    w_output += act_hidden.T.dot(dZ_output)
    
    dH = dZ_output.dot(w_output.T) * sigmoid_prime(act_hidden)
    w_hidden += X.T.dot(dH)

# Test the trained model
X_test = X[1]  # [0, 1]
act_hidden_test = sigmoid(np.dot(X_test, w_hidden))
output_test = sigmoid(np.dot(act_hidden_test, w_output))

print("Test input:", X_test)
print("Predicted output:", np.round(output_test))
