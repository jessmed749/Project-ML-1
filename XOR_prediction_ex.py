#create a Neural Network (NN) that can properly predict
# values from the XOR function.

import numpy as np

# Hyperparameters
epochs = 60000  # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 3, 1  # Layer sizes
LR = 0.1  # Learning rate

# XOR input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights with random values
w_hidden = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))    # Input features (4 samples, 2 features each)
w_output = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))   # Expected outputs

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return x * (1 - x)

# Training the neural network
for epoch in range(epochs):
    # Forward pass
    act_hidden = sigmoid(np.dot(X, w_hidden))   # Hidden layer activations
    output = sigmoid(np.dot(act_hidden, w_output))  # Output layer result
    
    # Calculate error
    error = y - output
    
    if epoch % 5000 == 0:
        print(f'Epoch {epoch}: error sum {np.sum(error)}')  # Monitor training progress
    
    # Backward pass
    dZ_output = error * LR * sigmoid_prime(output)  # Gradient for output layer
    w_output += act_hidden.T.dot(dZ_output) # Update output weights
    
    dH = dZ_output.dot(w_output.T) * sigmoid_prime(act_hidden)  # Gradient for hidden layer
    w_hidden += X.T.dot(dH) # Update hidden weights

# Test the trained model
X_test = X[1]  # Example input: [0, 1]
act_hidden_test = sigmoid(np.dot(X_test, w_hidden))
output_test = sigmoid(np.dot(act_hidden_test, w_output))

print("Test input:", X_test)
print("Predicted output:", np.round(output_test))
