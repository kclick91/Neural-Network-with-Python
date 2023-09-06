import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Generate some sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
print("Sample data.")
print(X)
print(y)

# Initialize weights and biases for the neural network
# Save
# input_size = 2
# hidden_size = 4
# output_size = 1
# End Save
input_size = int(input("Enter the input size."))#Example is 2 so enter 2 for the example
hidden_size = int(input("Enter the hidden size."))#Example is 4 so enter 4 for the example
output_size = int(input("Enter the output size."))#Example is 1 so enter 1 for the example



# Loop through input and output size user input


np.random.seed(42)  # For reproducibility
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
bias_output = np.zeros((1, output_size))

#Save
# learning_rate = 0.1
# epochs = 10000
#End Save

learning_rate = float(input("Enter the learning rate."))
epochs = int(input("Enter the number of epochs."))


# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)

    # Calculate the loss
    loss = y - output_layer_output

    # Backpropagation
    d_output = loss * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Evaluate the trained model
hidden_layer = sigmoid(np.dot(X, weights_input_hidden) + bias_hidden)
predicted_output = sigmoid(np.dot(hidden_layer, weights_hidden_output) + bias_output)

print("Predicted Output:")
print(predicted_output)

