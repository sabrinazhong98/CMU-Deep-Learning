"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

#sys.path.append('mytorch')
sys.path.append('C:/Users/zhong/Desktop/CMU/Academic/DeepLearning/hw/hw1/main/handout/mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *

class MLP(object):

    
  

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        
        
        self.size_list= [input_size]
        self.size_list.extend(hiddens)
        self.size_list.append(output_size)
        
        #now get the the pairs
        pairs = zip(self.size_list[:len(self.size_list)-1], self.size_list[1:])
        
        #add all the linear layers
        self.linear_layers = [Linear(inf, ouf, weight_init_fn, bias_init_fn) for inf, ouf in pairs]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn == True:
            self.bn_layers = [BatchNorm(val) for i, val in enumerate(self.size_list[1:]) if i < self.bn]


    def forward(self, x):
       
        # Complete the forward pass through your entire MLP.
        
        for index, l in enumerate(self.linear_layers):
            
            val = l.forward(x)
            
            
            if self.bn == True and index < self.num_bn_layers :
                
                if self.train_mode == True:
                   val = self.bn_layers[index].forward(val)
                else:
                   val = self.bn_layers[index].forward(val, eval = True) 
        
            #pass activation
            if index > 0 or index < len(self.linear_layers)-1:
                
                val = self.activations[index].forward(val)
            
            x = val
            
        self.output = val
        return val
     
    
    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        
        for l in self.linear_layers:
    
            l.dW.fill(0.0)
            l.db.fill(0.0)
        
        if self.bn == True:
            for b in self.bn_layers:
                b.dbeta.fill(0.0)
                b.dgamma.fill(0.0)

    
    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)
      


        for i in range(len(self.linear_layers)):
            
            #update the batchnorm layer
            if self.bn == True and i < self.num_bn_layers:
                self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr * self.bn_layers[i].dgamma 
                self.bn_layers[i].beta = self.bn_layers[i].beta - self.lr*self.bn_layers[i].dbeta
            
            #the intermediate value
            self.linear_layers[i].momentum_W = self.momentum*self.linear_layers[i].momentum_W - self.lr * self.linear_layers[i].dW
            self.linear_layers[i].momentum_b = self.momentum*self.linear_layers[i].momentum_b - self.lr * self.linear_layers[i].db
       
            self.linear_layers[i].W = self.linear_layers[i].W + self.linear_layers[i].momentum_W
            self.linear_layers[i].b = self.linear_layers[i].b + self.linear_layers[i].momentum_b
 
    def backward(self, labels):

        
        output = self.output
        self.criterion.forward(output, labels)
        dy = self.criterion.derivative()
    
        
        #back through activation functions
        for i in range(self.nlayers-1, -1,-1):
            dz_current = dy * self.activations[i].derivative() 
            
                
            if i >= self.num_bn_layers:   
                dy_current = self.linear_layers[i].backward(dz_current)
        
            
            elif self.bn and i < self.num_bn_layers:
                dz_current = self.bn_layers[i].backward(dz_current)
                dy_current = self.linear_layers[i].backward(dz_current)
                
            dy = dy_current
                
    
    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

#This function does not carry any points. You can try and complete this function to train your network.
def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):

            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):

            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented
