# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 18:29:50 2019

@author: marks

Script to create a NN for recognising a checkerboard pattern in a 2x2 grid, as 
a simple test for the NN class. 

The input signals [1, 2, 3, 4] represent the 2x2 grid, given by: [[1, 3],
                                                                  [4, 2]]
This is done to make the NN easy to design. 

The structure of the NN is inspired by: 
    
https://www.youtube.com/watch?v=JeVDjExBf7Y&list=LLptkJz2aE0zu2bQc8b2zRnQ&index=4&t=0s

"""

import math
import numpy as np

list_of_weights = []

m = np.array([[-1, -1, 0, 0, 0.5], [1, 1, 0, 0, -1.5], [0, 0, -1, -1, 0.5], [0, 0, 1, 1, -1.5], [0, 0, 0, 0, math.inf]])
list_of_weights.append(m)

m = np.array([[0, 1, 1, 0, -1.5], [1, 0, 0, 1, -1.5], [0, 0, 0, 0, math.inf]])
list_of_weights.append(m)

m = np.array([[1, 1, -0.5], [0, 0, math.inf]])
list_of_weights.append(m)

nn = NeuralNetwork(nn = list_of_weights, af = 'step')
print(nn)

print('Enter 2x2 grid, as index of [1, 2, 3, 4] corresponding to grid: ')
print('[[1, 3], ')
print(' [4, 2]]\n')
for i in range(3):
    string_2x2 = input('Enter 2x2 grid "a,b,c,d": ')
    grid_2x2 = [int(c) for c in string_2x2.split(',')]
    print('Checkerboard? = ', nn.predict(grid_2x2), '\n')