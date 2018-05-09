from __future__ import print_function
import numpy as np

def nonlin(x,deriv=False): 
    if(deriv==True):
        return x*(1-x)
    return 1/(1+(np.exp(-x)))

# 3 input XOR gate
inpt = np.asarray([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]);
desired_out = np.asarray([[1],[0],[0],[1],[0],[1],[1],[0]]);
np.random.seed(1)
 #Random weight initialization
weights0 = 2*np.random.random((3,8)) - 1 
weights1 = 2*np.random.random((8,1)) - 1

#Iteration
for j in range(600000): 
    
    layer0 = inpt
    layer1 = nonlin(np.dot(layer0,weights0))
    layer2 = nonlin(np.dot(layer1,weights1))
    
    layer2_error = desired_out - layer2
    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(layer2_error))))#Error at each 1000th iteration
    
    #Learning using backpropagation
    layer2_delta = layer2_error*nonlin(layer2,deriv=True) 
    layer1_error = layer2_delta.dot(weights1.T)
    layer1_delta = layer1_error*nonlin(layer1,deriv=True)
    
    weights1 += layer1.T.dot(layer2_delta)*0.1
    weights0 += layer0.T.dot(layer1_delta)*0.1
    
#Actual output
out = nonlin(np.dot(nonlin(np.dot(inpt,weights0)),weights1))