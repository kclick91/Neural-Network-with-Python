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

# Initialize weights and biases for the neural network
# Save
# Default
input_size = 2
hidden1_size = 4
hidden2_size = 4
output_size = 1
# End Save

input_size = int(input("Enter the input size."))#Example is 2 so enter 2 for the example
hidden1_size = int(input("Enter the first hidden size."))#Example is 4 so enter 4 for the example
hidden2_size = int(input("Enter the second hidden size."))
output_size = int(input("Enter the output size."))#Example is 1 so enter 1 for the example




# What is the chance of each team winning based solely on the inning and score in that inning?

# START MLB game stats
# EARLIER_GAMES = np.array ([[[36, 9, 11, 8, 6, 12], [34, 2, 4, 2, 4, 9]], [[34, 4, 7, 4, 2, 9], [34, 8, 13, 8, 3, 12]]])
# Input size is 3 for the scores at the end of each inning and the inning
# Output size is 1, home team(0) or away team(1) wins

# Games of 9/9/2023
# Brewers v Yankees, Mets v Twins, Diamondbacks vs Cubs, Royals vs Blue Jays, Dodgers vs Nationals, White Sox vs Tiger, Cardinals vs Reds, Athletics vs Rangers, Padres vs Astros
# Pirates v Braves, Rockies vs Giants, Guardians vs Angels 
# Score and inning without consideration of anything about the specific teams
EARLIER_GAMES = np.array([[0,0,1], [0,0,2], [0,0,3], [2,2,4],[2,2,5],[2,2,6], [2,2,7], [5,2,8], [9,2,9], [2,0,1], [2,2,2], [2,3,3], [2,3,4], [2,3,5], [2,3,6], 
[2,7,7], [4,8,8], [4,8,9], [0,0,1], [0,0,2], [0,1,3], [0,1,4],[1,1,5],[1,1,6],[1,1,7], [1,1,8], [1,1,9], [2,1,10], [0,0,1], [0,0,2], [0,0,3], [0,1,4], [1,3,5], 
[1,4,6], [1,5,7], [1,5,8], [1,5,9], [1,1,1], [1,1,2], [1,1,3], [1,1,4], [1,3,5], [1,3,6], [2,5,7], [4,5,8],[5,5,9], [6,6,10], [6,7,11], [0,0,1], [0,1,2], [0,1,3],
[0,1,4], [0,1,5], [0,2,6], [0,3,7], [1,3,8], [1,3,9], [1,0,1], [1,3,2], [3,3,3], [4,3,4], [4,3,5], [4,3,6], [4,3,7], [4,3,8], [4,3,9], [0,0,1], [0,0,2],[0,0,3],
[0,0,4],[0,0,5],[0,2,6],[2,3,7],[2,3,8],[2,3,9],[0,0,1],[0,0,2],[0,1,3],[4,2,4],[4,7,5],[5,7,6],[5,7,7],[5,7,8],[5,7,9],[0,0,1], [0,0,2],[3,1,3],[3,3,4],
[6,3,5],[7,3,6],[7,4,7],[7,4,8],[8,4,9],[0,2,1],[0,2,2],[0,2,3],[0,6,4],[0,6,5],[0,9,6],[1,9,7],[1,9,8],[1,9,9],[1,2,1],[1,2,2],[2,3,3],[2,4,4],[2,4,5],[2,4,6],
[2,4,7],[2,6,8],[2,6,9]])

EARLIER_GAMES_RESULTS = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],
 [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],
 [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],
 [0],[0],[0],[0],[0],[0],[0],[0],[0]])



# END MLB game stats


# TEST MLB game
# 9/9/2023 Mariners vs Rays at end of 3rd inning, end of 5th ending and end of 8th inning
TEST_GAMES = [[4,1,3], [4,3,5], [5,5,8]]
#9/9/2023 Orioles vs Red Sox at end of 3rd, 6th, and eighth AND 9/9/2023 Marlins vs Phillies at end of 3rd, sixth and eigth
TEST_GAMES_TWO = [[5,2,3],[9,6,6],[12,9,8],[0,5,3],[4,8,6],[4,8,8]]

# END TEST MLB game

# TEST MADE UP GAMES
TEST_MADEUPGAMES = [[0,0,1],[0,0,2],[0,0,3], [0,0,4], [0,0,5], [0,0,6], [0,0,7], [0,0,8], [0,0,9], [4,4,3], [4,4,5], [4,4,7], [10,0,1], [10,0,5], [10,0,7],
[0,10,1], [0,10,3], [0,10,5], [0,10,7], [1,3,9],[4,5,9],[10,1,9]]



# Loop through input and output size user input


#Save
# learning_rate = 0.1
# epochs = 10000
#End Save


#DEFAULT
learning_rate = 0.1
epochs = 10000


# Evaluate the trained model
# hidden1_layer = sigmoid(np.dot(TEST_GAMES_TWO, weights_input_hidden1) + bias_hidden1)
# hidden2_layer = sigmoid(np.dot(hidden1_layer, weights_hidden1_hidden2) + bias_hidden2)
# predicted_output = sigmoid(np.dot(hidden2_layer, weights_hidden2_output) + bias_output)

# print("Predicted Output:")
# print(predicted_output)

def EnterLearning():
    learning_rate = float(input("Enter the learning rate."))
    epochs = int(input("Enter the number of epochs."))

def train(EARLY, EARLYRESULTS, LATER):
    np.random.seed(42)  # For reproducibility
    weights_input_hidden1 = np.random.uniform(size=(input_size, hidden1_size))
    bias_hidden1 = np.zeros((1, hidden1_size))
    weights_hidden1_hidden2 = np.random.uniform(size=(hidden1_size, hidden2_size))
    bias_hidden2 = np.zeros((1, hidden2_size))
    weights_hidden2_output = np.random.uniform(size=(hidden2_size, output_size))
    bias_output = np.zeros((1, output_size))
    for epoch in range(epochs):
        # Forward propagation
        hidden1_layer_input = np.dot(EARLY, weights_input_hidden1) + bias_hidden1
        hidden1_layer_output = sigmoid(hidden1_layer_input)

        hidden2_layer_input = np.dot(hidden1_layer_output, weights_hidden1_hidden2) + bias_hidden2
        hidden2_layer_output = sigmoid(hidden2_layer_input)

        output_layer_input = np.dot(hidden2_layer_output, weights_hidden2_output) + bias_output
        output_layer_output = sigmoid(output_layer_input)

        # Calculate the loss
        loss = EARLYRESULTS - output_layer_output

        # Backpropagation
        d_output = loss * sigmoid_derivative(output_layer_output)
        error_hidden2_layer = d_output.dot(weights_hidden2_output.T)
        d_hidden2_layer = error_hidden2_layer * sigmoid_derivative(hidden2_layer_output)
        error_hidden1_layer = d_hidden2_layer.dot(weights_hidden1_hidden2.T)
        d_hidden1_layer = error_hidden1_layer * sigmoid_derivative(hidden1_layer_output)

        # Update weights and biases
        weights_hidden2_output += hidden2_layer_output.T.dot(d_output) * learning_rate
        bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        weights_hidden1_hidden2 += hidden1_layer_output.T.dot(d_hidden2_layer) * learning_rate
        bias_hidden2 += np.sum(d_hidden2_layer, axis=0, keepdims=True) * learning_rate
        weights_input_hidden1 += EARLY.T.dot(d_hidden1_layer) * learning_rate
        bias_hidden1 += np.sum(d_hidden1_layer, axis=0, keepdims=True) * learning_rate
    
    # Evaluate the trained model
    hidden1_layer = sigmoid(np.dot(LATER, weights_input_hidden1) + bias_hidden1)
    hidden2_layer = sigmoid(np.dot(hidden1_layer, weights_hidden1_hidden2) + bias_hidden2)
    predicted_output = sigmoid(np.dot(hidden2_layer, weights_hidden2_output) + bias_output)
    
    print("Predicted Output:")
    print(predicted_output)


def MLBBoxScore():
    
    for i in range(10):
        print("Enter the away score, home score, and inning. 0 means home is predicted to win and 1 means away is predicted to win.")
        away = int(input("Away."))
        home = int(input("Home."))
        inning = int(input("Inning."))
        ARR = np.array([away, home, inning])
        hidden1_layerEntered = sigmoid(np.dot(ARR, weights_input_hidden1) + bias_hidden1)
        hidden2_layerEntered = sigmoid(np.dot(hidden1_layerEntered, weights_hidden1_hidden2) + bias_hidden2)
        predicted_outputEntered = sigmoid(np.dot(hidden2_layerEntered, weights_hidden2_output) + bias_output)
        print(predicted_outputEntered)
      
EnterLearning() 
train(EARLIER_GAMES, EARLIER_GAMES_RESULTS, TEST_GAMES_TWO)
MLBBoxScore()



