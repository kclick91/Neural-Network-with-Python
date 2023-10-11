import numpy as np

# START ACTIVATION FUNCTIONS
# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


# END ACTIVATION FUNCTIONS
# Generate some sample data
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases for the neural network
# Save
# Default
# input_size = 2
# hidden1_size = 4
# hidden2_size = 4
# output_size = 1
# End Savehidd


input_size = int(input("Enter the input size."))
hidden1_size = int(input("Enter the first hidden size."))
hidden2_size = int(input("Enter the second hidden size."))
output_size = int(input("Enter the output size."))




# What is the chance of each team winning based solely on the inning and score in that inning?

# START MLB game stats
# EARLIER_GAMES = np.array ([[[36, 9, 11, 8, 6, 12], [34, 2, 4, 2, 4, 9]], [[34, 4, 7, 4, 2, 9], [34, 8, 13, 8, 3, 12]]])
# Input size is 3 for the scores at the end of each inning and the inning
# Output size is 1, home team(0) or away team(1) wins

# Games of 9/9/2023
# Brewers v Yankees, Mets v Twins, Diamondbacks vs Cubs, Royals vs Blue Jays, Dodgers vs Nationals, White Sox vs Tiger, Cardinals vs Reds, Athletics vs Rangers, Padres vs Astros
# Pirates v Braves, Rockies vs Giants, Guardians vs Angels
# Games of 9/19/2023
# Angels vs Rays, Twins vs Reds, Mets vs Marlins, White Sox vs Nationals, Blue Jays vs Yankees, Phillies vs Braves, Guardians vs Royals, Pirates vs Cubs, Brewers vs Cardinals
# Games of 9/9/2023
# Brewers vs Yankees
# Games of 7/7/2023
# Blue Jays vs Tigers, Braves vs Rays, Phillies vs Marlins
# Score and inning without consideration of anything about the specific teams
EARLIER_GAMES = np.array([[0,0,1], [0,0,2], [0,0,3], [2,2,4],[2,2,5],[2,2,6], [2,2,7], [5,2,8], [9,2,9], [2,0,1], [2,2,2], [2,3,3], [2,3,4], [2,3,5], [2,3,6], 
[2,7,7], [4,8,8], [4,8,9], [0,0,1], [0,0,2], [0,1,3], [0,1,4],[1,1,5],[1,1,6],[1,1,7], [1,1,8], [1,1,9], [2,1,10], [0,0,1], [0,0,2], [0,0,3], [0,1,4], [1,3,5], 
[1,4,6], [1,5,7], [1,5,8], [1,5,9], [1,1,1], [1,1,2], [1,1,3], [1,1,4], [1,3,5], [1,3,6], [2,5,7], [4,5,8],[5,5,9], [6,6,10], [6,7,11], [0,0,1], [0,1,2], [0,1,3],
[0,1,4], [0,1,5], [0,2,6], [0,3,7], [1,3,8], [1,3,9], [1,0,1], [1,3,2], [3,3,3], [4,3,4], [4,3,5], [4,3,6], [4,3,7], [4,3,8], [4,3,9], [0,0,1], [0,0,2],[0,0,3],
[0,0,4],[0,0,5],[0,2,6],[2,3,7],[2,3,8],[2,3,9],[0,0,1],[0,0,2],[0,1,3],[4,2,4],[4,7,5],[5,7,6],[5,7,7],[5,7,8],[5,7,9],[0,0,1], [0,0,2],[3,1,3],[3,3,4],
[6,3,5],[7,3,6],[7,4,7],[7,4,8],[8,4,9],[0,2,1],[0,2,2],[0,2,3],[0,6,4],[0,6,5],[0,9,6],[1,9,7],[1,9,8],[1,9,9],[1,2,1],[1,2,2],[2,3,3],[2,4,4],[2,4,5],[2,4,6],
[2,4,7],[2,6,8],[2,6,9],[0,2,1],[0,2,2],[0,2,3],[1,2,4],[1,2,5],[1,2,6],[1,2,7],[2,6,8],[2,6,9],[0,0,1],[1,0,2],[1,0,3],[2,0,4],[2,0,5],[4,0,6],[6,0,7],[7,0,8],[7,0,9],
[0,0,1],[0,0,2],[1,1,3],[1,1,4],[1,2,5],[1,3,6],[1,3,7],[1,3,8],[3,4,9],[0,0,1],[0,0,2],[0,0,3],[1,0,4],[1,0,5],[1,1,6],[2,4,7],[2,4,8],[3,4,9],[1,1,1],[1,1,2],[1,1,3],
[2,1,4],[4,1,5],[4,1,6],[4,1,7],[4,1,8],[7,1,9],[0,1,1],[0,1,2],[0,1,3],[0,3,4],[0,7,5],[3,9,6],[3,9,7],[3,9,8],[3,9,9],[0,2,1],[0,2,2],[0,4,3],[0,4,4],[2,7,5],
[4,7,6],[5,7,7],[6,7,8],[6,7,9],[0,2,1],[0,4,2],[1,5,3],[1,5,4],[1,5,5],[1,6,6],[1,6,7],[1,14,8],[1,14,9],[0,2,1],[0,2,2],[1,2,3],[5,2,4],[5,3,5],[5,3,6],[5,3,7],[6,3,8],
[7,3,9],[0,0,1],[0,0,2],[0,0,3],[2,2,4],[2,2,5],[2,2,6],[2,2,7],[5,2,8],[9,2,9],[0,0,1],[0,0,2],[1,1,3],[7,1,4],[7,1,5],[7,1,6],[7,1,7],[7,2,8],[12,2,9],[0,1,1],[0,1,2],
[0,1,3],[2,1,4],[2,1,5],[2,1,6],[2,1,7],[2,1,8],[2,1,9],[0,0,1],[0,2,2],[0,2,3],[0,3,4],[0,3,5],[1,3,6],[1,3,7],[1,3,8],[4,3,9]])

EARLIER_GAMES_RESULTS = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],
 [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],
 [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],
 [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],
 [0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
 [0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],
 [1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])


# END MLB game stats

# START PREDICTING HOME RUNS
# Based on ranges

# END PREDICTING HOME RUNS

# 

#




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

# learning_rate = float(input("Enter the learning rate."))
# epochs = int(input("Enter the number of epochs."))

np.random.seed(42)  # For reproducibility
weights_input_hidden1 = np.random.uniform(size=(input_size, hidden1_size))
bias_hidden1 = np.zeros((1, hidden1_size))
weights_hidden1_hidden2 = np.random.uniform(size=(hidden1_size, hidden2_size))
bias_hidden2 = np.zeros((1, hidden2_size))
weights_hidden2_output = np.random.uniform(size=(hidden2_size, output_size))
bias_output = np.zeros((1, output_size))
print("Weights: ", weights_input_hidden1)


resultsArr = np.zeros(shape=(10,1))
print("Results: " , resultsArr)
def train(EARLY, EARLYRESULTS, LATER):
    
    for epoch in range(epochs):
        # Forward propagation
        hidden1_layer_input = np.dot(EARLY, globals()["weights_input_hidden1"]) + globals()["bias_hidden1"]
        hidden1_layer_output = sigmoid(hidden1_layer_input)
        hidden2_layer_input = np.dot(hidden1_layer_output, globals()["weights_hidden1_hidden2"]) + globals()["bias_hidden2"]

        hidden2_layer_output = sigmoid(hidden2_layer_input)
        output_layer_input = np.dot(hidden2_layer_output, globals()["weights_hidden2_output"]) + globals()["bias_output"]
        output_layer_output = sigmoid(output_layer_input)

        # Calculate the loss
        loss = EARLYRESULTS - output_layer_output

        # Backpropagation
        d_output = loss * sigmoid_derivative(output_layer_output)
        error_hidden2_layer = d_output.dot(globals()["weights_hidden2_output"].T)
        d_hidden2_layer = error_hidden2_layer * sigmoid_derivative(hidden2_layer_output)
        error_hidden1_layer = d_hidden2_layer.dot(globals()["weights_hidden1_hidden2"].T)
        d_hidden1_layer = error_hidden1_layer * sigmoid_derivative(hidden1_layer_output)

        # Update weights and biases
        globals()["weights_hidden2_output"] += hidden2_layer_output.T.dot(d_output) * learning_rate
        globals()["bias_output"] += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        globals()["weights_hidden1_hidden2"] += hidden1_layer_output.T.dot(d_hidden2_layer) * learning_rate
        globals()["bias_hidden2"] += np.sum(d_hidden2_layer, axis=0, keepdims=True) * learning_rate
        globals()["weights_input_hidden1"] += EARLY.T.dot(d_hidden1_layer) * learning_rate
        globals()["bias_hidden1"] += np.sum(d_hidden1_layer, axis=0, keepdims=True) * learning_rate
    
    # Evaluate the trained model
    hidden1_layer = sigmoid(np.dot(LATER, globals()["weights_input_hidden1"]) + globals()["bias_hidden1"])
    hidden2_layer = sigmoid(np.dot(hidden1_layer, globals()["weights_hidden1_hidden2"]) + globals()["bias_hidden2"])
    predicted_output = sigmoid(np.dot(hidden2_layer, globals()["weights_hidden2_output"]) + globals()["bias_output"])
    
    print("Predicted Output:")
    print(predicted_output)
    
def TrainSame(EARLY, EARLYRESULTS, home, away, inning, learning_rate, epochs):
    for epoch in range(epochs):
        # Forward propagation
        hidden1_layer_input = np.dot(EARLY, globals()["weights_input_hidden1"]) + globals()["bias_hidden1"]
        hidden1_layer_output = sigmoid(hidden1_layer_input)
        hidden2_layer_input = np.dot(hidden1_layer_output, globals()["weights_hidden1_hidden2"]) + globals()["bias_hidden2"]

        hidden2_layer_output = sigmoid(hidden2_layer_input)
        output_layer_input = np.dot(hidden2_layer_output, globals()["weights_hidden2_output"]) + globals()["bias_output"]
        output_layer_output = sigmoid(output_layer_input)

        # Calculate the loss
        loss = EARLYRESULTS - output_layer_output

        # Backpropagation
        d_output = loss * sigmoid_derivative(output_layer_output)
        error_hidden2_layer = d_output.dot(globals()["weights_hidden2_output"].T)
        d_hidden2_layer = error_hidden2_layer * sigmoid_derivative(hidden2_layer_output)
        error_hidden1_layer = d_hidden2_layer.dot(globals()["weights_hidden1_hidden2"].T)
        d_hidden1_layer = error_hidden1_layer * sigmoid_derivative(hidden1_layer_output)

        # Update weights and biases
        globals()["weights_hidden2_output"] += hidden2_layer_output.T.dot(d_output) * learning_rate
        globals()["bias_output"] += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        globals()["weights_hidden1_hidden2"] += hidden1_layer_output.T.dot(d_hidden2_layer) * learning_rate
        globals()["bias_hidden2"] += np.sum(d_hidden2_layer, axis=0, keepdims=True) * learning_rate
        globals()["weights_input_hidden1"] += EARLY.T.dot(d_hidden1_layer) * learning_rate
        globals()["bias_hidden1"] += np.sum(d_hidden1_layer, axis=0, keepdims=True) * learning_rate
    
    # Evaluate the trained model
    LATER = [[away, home, inning]]
    hidden1_layer = sigmoid(np.dot(LATER, globals()["weights_input_hidden1"]) + globals()["bias_hidden1"])
    hidden2_layer = sigmoid(np.dot(hidden1_layer, globals()["weights_hidden1_hidden2"]) + globals()["bias_hidden2"])
    predicted_output = sigmoid(np.dot(hidden2_layer, globals()["weights_hidden2_output"]) + globals()["bias_output"])
    
    print("Predicted Output:")
    print(predicted_output)
    return predicted_output


def TrainSameColl(EARLY, EARLYRESULTS, homeScore, awayScore, inningInp):
    pointOne = TrainSame(EARLY, EARLYRESULTS, homeScore, awayScore, inningInp, 0.1, 10000)
    pointOneFive = TrainSame(EARLY, EARLYRESULTS, homeScore, awayScore, inningInp, 0.15, 10000)
    pointTwo = TrainSame(EARLY, EARLYRESULTS, homeScore, awayScore, inningInp, 0.2, 10000)
    pointTwoFive = TrainSame(EARLY, EARLYRESULTS, homeScore, awayScore, inningInp, 0.25, 10000)
    pointThree = TrainSame(EARLY, EARLYRESULTS, homeScore, awayScore, inningInp, 0.3, 10000)
    pointThreeFive = TrainSame(EARLY, EARLYRESULTS, homeScore, awayScore, inningInp, 0.35, 10000)
    pointFour = TrainSame(EARLY, EARLYRESULTS, homeScore, awayScore, inningInp, 0.4, 10000)
    collectionTrainSame = np.array([pointOne, pointOneFive, pointTwo, pointTwoFive, pointThree, pointThreeFive, pointFour])
    
    #print(collectionTrainSame)
    
    compareFile = open("CompareFile.txt","a")
    for i in range(7):
        compareFile.write("Away Home Inning:  ")
        compareFile.write(str(awayScore))
        compareFile.write("   ")
        compareFile.write(str(homeScore))
        compareFile.write("   ")
        compareFile.write(str(inningInp))
        compareFile.write("   ")
        
        compareFile.write("Hidden Layer One Size:   ")
        compareFile.write(str(globals()["hidden1_size"]))
        compareFile.write("  ")
        compareFile.write("Hidden Layer Two Size:   ")
        compareFile.write(str(globals()["hidden2_size"]))
        compareFile.write("  ")
        compareFile.write("Learning Rate:   ")
        compareFile.write(str(0.1 + (0.05 * i)))
        compareFile.write("  ")
        compareFile.write("Epochs:   ")
        compareFile.write(str(globals()["epochs"]))
        compareFile.write("  ")
        compareFile.write("\n")
    compareFile.write("Prediction:   ")
    compareFile.write("\n")
    compareFile.write(str(collectionTrainSame))
    compareFile.write("\n")
    compareFile.close()

    return collectionTrainSame;
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
        resultFile = open("resFile.txt","a")
        resultFile.write("Away Home Inning:  ")
        resultFile.write(str(away))
        resultFile.write("   ")
        resultFile.write(str(home))
        resultFile.write("   ")
        resultFile.write(str(inning))
        resultFile.write("   ")
        
        resultFile.write("Hidden Layer One Size:   ")
        resultFile.write(str(globals()["hidden1_size"]))
        resultFile.write("  ")
        resultFile.write("Hidden Layer Two Size:   ")
        resultFile.write(str(globals()["hidden2_size"]))
        resultFile.write("  ")
        resultFile.write("Learning Rate:   ")
        resultFile.write(str(globals()["learning_rate"]))
        resultFile.write("  ")
        resultFile.write("Epochs:   ")
        resultFile.write(str(globals()["epochs"]))
        resultFile.write("  ")
        resultFile.write("Prediction:   ")
        resultFile.write(str(predicted_outputEntered))
        resultFile.write("\n")
        resultFile.close()
        
        
# train(EARLIER_GAMES, EARLIER_GAMES_RESULTS, TEST_GAMES_TWO)

#MLB scores on 




#MLBBoxScore()

# Provide a comparison of learning rate, epochs, and size of hidden layers
# Currently compares a learning rate of 0.1, 0.15, 0.2, 0.25
TrainSameColl(EARLIER_GAMES, EARLIER_GAMES_RESULTS, 1,0,2)

TrainSameColl(EARLIER_GAMES, EARLIER_GAMES_RESULTS,1,1,3)

TrainSameColl(EARLIER_GAMES, EARLIER_GAMES_RESULTS,6,1,3)

TrainSameColl(EARLIER_GAMES,EARLIER_GAMES_RESULTS,6,1,4)


# New neural network with the array of comparisons as input
