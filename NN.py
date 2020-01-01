# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:54:25 2019

@author: marks
"""

# Number of columns = number of input nodes
# Number of rows = number of output nodes
# Number of signals = number of input nodes
# Input list [2,3,2] which is 2 input, 3 hidden, 2 output nodes, giving:
# list of [3x2 matrix, 2x3 matrix] of weights (initialised as zeros)
# Input any list of length greater than 1, with int entries larger than zero

# - Bias nodes: 
#   - They are connected to hidden/output layers. They are each an 
#  'input' node and hence contribute to an extra signal and column. 
#   - So for bias nodes, add an extra row and column, with the weights from the 
#   input bias node set accordingly and the weights to the output bias node 
#   set to (0 0 ... 0 Inf) to guarantee a signal of 1. Also, add an extra signal 
#   which is always set to 1 in the input, and remove it from the output.  

# To do: 
# - Create back_prop method
# - Create neat method

import math
import numpy as np

def sigmoid(in_arr):
    '''
    Function to apply the sigmoid function to in_arr to get out_arr 
    
    Input: in_arr (input numpy array)
    
    Output: out_arr (output numpy array)
    '''
    out_arr = 1.0 / (1.0 + np.exp(-1.0 * in_arr))
    return out_arr

def binary_step(in_arr):
    '''
    Function to apply the binary step function to in_arr
    
    Input: in_arr (input numpy array)
    
    Output: in_arr (transformed numpy array to output)
    '''
    in_arr[in_arr >= 0] = 1
    in_arr[in_arr < 0] = 0
    
    return in_arr

class NeuralNetwork(object):
    '''
    An object which stores a NN, which can be set manually, trained or tested
    
    Each instance of NN is represented as a list of matrices where:
    - Each list entry represents a set of connections between two layers
    - The rows of each matrix represent the 'output' nodes
    - The columns of each matrix represent the 'input' nodes
    - The entries of each matrix are the weights connecting two nodes
    '''
     
    def __init__(self, **kwargs):
        '''
        Initialise by creating a NN, with the structure given at input and the 
        weights all initialised to input values/zero. 
        
        Input either:
        
        - A list containing the matrices of weights between nodes, including 
          the bias nodes. So the full NN. 
        - A list containing the number of nodes per layer, with the first 
        entry representing layer 1, the second entry representing layer 2, etc.
        This does not include the bias nodes, which are added automatically. 
        '''
        
        # NN
        if 'nn' in kwargs:
            self.nn = kwargs['nn']
        elif 'npl' in kwargs:
            nodes_per_layer = kwargs['npl']
            
            # Input Validation (to do)
            
            # For each set of connections between layers, create the matrix
            n_connections = len(nodes_per_layer) - 1
            self.nn = []
            for i in range(n_connections):
                # +1 for bias nodes
                n_in_nodes = nodes_per_layer[i] + 1
                n_out_nodes = nodes_per_layer[i+1] + 1
                weights = np.zeros((n_out_nodes, n_in_nodes))
                weights[n_out_nodes - 1, n_in_nodes - 1] = math.inf
                self.nn.append(weights)
        else:
            self.nn = []
            print('Warning: Invalid input for NN, so NN is an empty list')
        
        # Activation Function
        if 'af' in kwargs:
            self.activation_function = kwargs['af']
        else:  # Default: sigmoid funciton
            self.activation_function = 'sigmoid'
    
    def __str__(self):
        '''
        Print Method
        '''
        out_nn = '\nActivation Function: ' + self.activation_function + '\n\n'
        for m in self.nn:
            out_nn += str(m) + '\n\n'
        return out_nn
    
    def predict(self, signals):
        '''
        Predict method, which applies the NN to the input signals to get the 
        predictions for the output
        
        Input: The NN object, signals (input list of signals)
        
        Output: signals (output list of signals)
        '''
        # Add bias signal
        signals.append(1)
        
        # Apply NN
        if self.activation_function == 'sigmoid':
            for m in self.nn:
                signals = sigmoid(np.dot(m, signals))
        elif self.activation_function == 'step':
            for m in self.nn:
                signals = binary_step(np.dot(m, signals))
        else:
            print('Error: Invalid activation function')
        
        # Remove bias signal and convert to python list
        signals = np.ndarray.tolist(signals)
        signals.pop()
        
        return signals
    
    
    
    
    
    