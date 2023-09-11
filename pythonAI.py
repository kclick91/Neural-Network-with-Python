import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Generate some sample data
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [1], [1], [0]])

# Sample data of number of hits for the mlb home team(0) and away team (1) and whether the home team(0) or away team(1) wins
X = np.array ([[8, 8], [5, 2], [11, 10], [8, 6], [13, 6], [3, 12]])
y = np.array([[0], [0], [0], [0], [0], [1]])
# print("Sample data.")
# print(X)
# print(y)




#Test later
Z = np.array([[8, 8], [1, 2], [1, 10], [1, 6], [13, 9], [2, 12]])

# Initialize weights and biases for the neural network
# Save
# input_size = 2
# hidden_size = 4
# output_size = 1
# End Save
input_size = int(input("Enter the input size."))#Example is 2 so enter 2 for the example
hidden_size = int(input("Enter the hidden size."))#Example is 4 so enter 4 for the example
output_size = int(input("Enter the output size."))#Example is 1 so enter 1 for the example


# What is the chance of each team winning based solely on the inning and score in that inning?

# START MLB game stats
# EARLIER_GAMES = np.array ([[[36, 9, 11, 8, 6, 12], [34, 2, 4, 2, 4, 9]], [[34, 4, 7, 4, 2, 9], [34, 8, 13, 8, 3, 12]]])
# Input size is 3 for the scores at the end of each inning and the inning
# Output size is 1, home team(0) or away team(1) wins

# Games of 9/9/2023
# Brewers v Yankees, Mets v Twins, Diamondbacks vs Cubs, Royals vs Blue Jays, Dodgers vs Nationals 
# Score and inning without consideration of anything about the specific teams
EARLIER_GAMES = np.array([[0,0,1], [0,0,2], [0,0,3], [2,2,4],[2,2,5],[2,2,6], [2,2,7], [5,2,8], [9,2,9], [2,0,1], [2,2,2], [2,3,3], [2,3,4], [2,3,5], [2,3,6], 
[2,7,7], [4,8,8], [4,8,9], [0,0,1], [0,0,2], [0,1,3], [0,1,4],[1,1,5],[1,1,6],[1,1,7], [1,1,8], [1,1,9], [2,1,10], [0,0,1], [0,0,2], [0,0,3], [0,1,4], [1,3,5], 
[1,4,6], [1,5,7], [1,5,8], [1,5,9], [1,1,1], [1,1,2], [1,1,3], [1,1,4], [1,3,5], [1,3,6], [2,5,7], [4,5,8],[5,5,9], [6,6,10], [6,7,11]])

EARLIER_GAMES_RESULTS = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],
 [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])



# END MLB game stats


# TEST MLB game
# 9/9/2023 Mariners vs Rays at end of 3rd inning, end of 5th ending and end of 8th inning
TEST_GAMES = [[4,1,3], [4,3,5], [5,5,8]]


# END TEST MLB game



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
    hidden_layer_input = np.dot(EARLIER_GAMES, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)

    # Calculate the loss
    loss = EARLIER_GAMES_RESULTS - output_layer_output

    # Backpropagation
    d_output = loss * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += EARLIER_GAMES.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Evaluate the trained model
hidden_layer = sigmoid(np.dot(TEST_GAMES, weights_input_hidden) + bias_hidden)
predicted_output = sigmoid(np.dot(hidden_layer, weights_hidden_output) + bias_output)

print("Predicted Output:")
print(predicted_output)

