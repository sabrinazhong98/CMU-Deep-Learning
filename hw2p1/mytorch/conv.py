# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.x = x
    
        batch_size = x.shape[0]
        in_channel = x.shape[1]
        input_size = x.shape[2]
        
        output_size = ((input_size - self.kernel_size)//self.stride) + 1
        output = np.zeros([batch_size, self.out_channel, output_size])
        
        for i in range(0, batch_size):
            for n in range(0,self.out_channel):
                for m in range(0, output_size):
                    low = m*self.stride
                    upper = m*self.stride + self.kernel_size
                    segment = x[i,:, low: upper]
                    if segment.shape[1] == self.kernel_size:
           
                        out = np.sum(segment*self.W[n])
                        output[i, n, m] = out
                        #add a bias term
                output[i,n] =  output[i,n] + self.b[n]
        return output



    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        batch_size = delta.shape[0]
        out_channel = delta.shape[1]
        output_size = delta.shape[2]
        
        in_channel = self.x.shape[1]
        input_size = self.x.shape[2]
        
        dx = np.zeros([batch_size, in_channel, input_size])
        for b in range(0,batch_size):
            for i in range(0,in_channel):
                 for o in range(0, output_size):
                 
                    low = o*self.stride
                    upper = o*self.stride + self.kernel_size
                  #  if upper <= input_size:
                
                    for p in range(low, upper):
                        
                            dx[b,i,p] += sum(self.W[out, i, p-low] * delta[b,out, o] for out in range(0,out_channel))
         
        #update dw
        self.dW = np.zeros(self.W.shape)
        
        
        for out in range(0, out_channel):
           
            for k in range(0, self.kernel_size):
                    
                for i in range(0, in_channel):
                    
                    for b in range(0, batch_size):
                
                        self.dW[out, i, k] += sum([self.x[b, i, o* self.stride+k] * delta[b, out, o] for o in range(0, output_size) ])
                      #  if o*self.stride+self.kernel_size<input_size
            
            
            
        
        self.db = np.sum(delta,axis = (0,2))
        return dx

class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
            
        
        """
        self.x = x
        batch_size = x.shape[0]
        in_channel = x.shape[1]
        width = x.shape[2]
        height = x.shape[3]
        
        
        output_size = ((width - self.kernel_size)//self.stride) + 1
        
        output = np.zeros([batch_size, self.out_channel, output_size, output_size])
        
        
        for b in range(0, batch_size):
            for out in range(0,self.out_channel):
        
                for m in range(0, output_size): #refers to vertically
                    for n in range(0, output_size): # refers to horizontally 
                    
                        
                        low = n * self.stride
                      
                        low2 = m * self.stride
                   
                        output[b,out,m,n] = np.sum( x[b, : ,low2 :low2+self.kernel_size, low:low+self.kernel_size] * self.W[out])
            output[b,out] +=   self.b[out]
      
        return output
        
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        
        batch_size = self.x.shape[0]
        in_channel = self.x.shape[1]
        width = self.x.shape[2]
        height = self.x.shape[3]
        output_size = ((width - self.kernel_size)//self.stride) + 1
        dx = np.zeros([batch_size, self.in_channel, width, height])
        dx = np.zeros([batch_size, self.in_channel, width, height])
        for b in range(0, batch_size):
            for i in range(0, in_channel):
                
                for m in range(0, output_size): #height
                    for n in range(0, output_size): #width
                        low = n*self.stride
                        upper = n*self.stride + self.kernel_size
                        
                        low2 = m*self.stride
                        upper2 = m*self.stride + self.kernel_size
                        for h in range(low2,upper2):
                            for k in range(low,upper):
                               
                                dx[b, i,h,k]  += sum([self.W[out, i, h-low2, k-low] * delta[b, out, m, n] for out in range(0, self.out_channel)])
                                        
     
        for out in range(0, self.out_channel):
            for k in range(0, self.kernel_size):
                for s in range(0,self. kernel_size):
                    for i in range(0, in_channel):
                        
                        for b in range(0, batch_size):
                            for m in range(0, output_size):
                                for n in range(0, output_size):
                                    low = n*self.stride
                                    upper = n*self.stride + self.kernel_size
                                    
                                    low2 = m*self.stride
                                    upper2 = m*self.stride + self.kernel_size
                                   
                                    self.dW[out, i, k, s] += self.x[b,i,low2+k, low+s] * delta[b,out, m,n]                             
        
        self.db = np.sum(np.sum(delta,axis = (0,2)), axis = 1)
                                        
        return dx
        

class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        
    
        return x.reshape(self.b, np.prod(self.c*self.w))

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
            
            
        """
        return np.reshape(delta,(self.b, self.c, self.w))
