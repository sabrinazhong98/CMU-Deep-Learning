# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)
    
    def forward(self, x, eval=False):
        self.x = x
        if eval == False:
    
            self.mean = np.mean(x, axis = 0)
            
            self.var = np.var(x, axis = 0)
            self.norm = (self.x - self.mean)/((self.var + self.eps)**(1/2))
            
    
            # Update running batch statistics
            self.running_mean = self.alpha * self.running_mean + (1-self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1-self.alpha) * self.var
            self.out = self.gamma *self.norm +self.beta
        else:
         
            self.norm = (self.x - self.running_mean) / ((self.running_var + self.eps)**(1/2))
            self.out = self.gamma *self.norm +self.beta
    
    
        
        return self.out


    def backward(self, delta):
   
        m = self.x.shape[0]
        
        
        self.dbeta = np.sum(delta, axis = 0, keepdims = True)
        self.dgamma = np.sum(delta*self.norm, axis = 0, keepdims = True)
        
        dx = delta * self.gamma
        dvar = -1/2 * np.sum(dx*(self.x-self.mean)*((self.var + self.eps)**(-3/2)), axis = 0)
        dmean = -np.sum(dx*((self.var + self.eps)**(-1/2)), axis = 0)-(2/m) * dvar*np.sum(self.x-self.mean, axis = 0)
        
        c1 = dx * ((self.var + self.eps)**(-1/2))
        c2 = dvar*(2/m)*(self.x-self.mean)
        c3 = dmean/m
        dx_out = c1 + c2 + c3
        
        print(-np.sum(dx/ (self.var + self.eps)**(1/2), axis = 0))
        

        return dx_out

    