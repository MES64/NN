# NN
To create an artificial neural network (NN) class from square one, and create NNs to 
solve problems like the iris dataset (using back propagation to train) or to play a 
game of asteroids (trained using NEAT).

#########################################################################################
2x2_diag_detect.py

To use:

1: Run NN.py

2: Run 2x2_diag_detect.py

3: Input numbers a,b,c,d from {0,1} (repeated 3 times)

4: Output either: 1.0 = yes for checkerboard, 0.0 = no

Script to create a NN for recognising a checkerboard pattern in a 2x2 grid, as 
a simple test for the NN class. 

The input signals [1, 2, 3, 4] represent the 2x2 grid, given by: [[1, 3],
                                                                  [4, 2]]
This is done to make the NN easy to design. 

The structure of the NN is inspired by: 
    
https://www.youtube.com/watch?v=JeVDjExBf7Y&list=LLptkJz2aE0zu2bQc8b2zRnQ&index=4&t=0s

##########################################################################################
NN.py

This script contains the NN class. So far it has:

-> init

Initialise by creating a NN, with the structure given at input and the 
weights all initialised to input values/zero. 

Input either:

- A list containing the matrices of weights between nodes, including 
the bias nodes. So the full NN. Prioritised over npl. E.g. nn = list_of_weights
- A list containing the number of nodes per layer, with the first 
entry representing layer 1, the second entry representing layer 2, etc.
This does not include the bias nodes, which are added automatically. 
E.g. npl = list_of_nodes_per_layer

Also, input the activation function. E.g. af = 'step' or af = 'sigmoid'

Example:

nnet = NeuralNetwork(nn = list_of_weights, af = 'step')

-> str

print(nnet) method

-> predict

Predict method, which applies the NN to the input signals to get the 
predictions for the output

Input: The NN object, signals (input list of signals). E.g. nnet.predict(signals_in)

Output: signals (output list of signals)

