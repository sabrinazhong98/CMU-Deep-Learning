import numpy as np


class CTC(object):
    """CTC class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument
        --------
        blank: (int, optional)
                blank label index. Default 0.

        """
        self.BLANK = BLANK

    def targetWithBlank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = 1)
                target output

        Return
        ------
        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks
        skipConnect: (np.array, dim = 1)
                    skip connections

        """
        
        multi = len(target)*2 +1
        extSymbols = [self.BLANK] * multi
        skipConnect = [self.BLANK] * multi
        

        # -------------------------------------------->

        # Your Code goes here
     
        j = 0
        for i in range(0,len(target)):
            extSymbols[j] = self.BLANK
            skipConnect[j] = 0
  
            j += 1
            
            extSymbols[j] = target[i]

            if i>0 and target[i] != target[i-1]:
                skipConnect[j] = 1
            else:
                skipConnect[j] = 0
            
            j += 1
        extSymbols[j] = self.BLANK
        skipConnect[j] = 0
        
            
        
        # <---------------------------------------------
        

        return extSymbols, skipConnect

    def forwardProb(self, logits, extSymbols, skipConnect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, channel))
                predict (log) probabilities

        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks

        skipConnect: (np.array, dim = 1)
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (output len, out channel))
                forward probabilities

        """
       # print('logits',logits)
        S, T = len(extSymbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->

        # Your Code goes here
        alpha[0,0] = logits[0, extSymbols[0]]
        alpha[0,1] = logits[0, extSymbols[1]]
        alpha[0,2:S] = 0
        for t in range(1, T):
            alpha[t,0] = alpha[t-1,0] * logits[t, extSymbols[0]]
            for i in range(1, S):
                alpha[t,i] = alpha[t-1,i-1] + alpha[t-1,i]
                if skipConnect[i]:
                    alpha[t,i] += alpha[t-1,i-2]
                alpha[t,i] *= logits[t, extSymbols[i]]
            
        # <---------------------------------------------
    
        return alpha

    def backwardProb(self, logits, extSymbols, skipConnect):
        """Compute backward probabilities.

        Input
        -----

        logits: (np.array, dim = (input_len, channel))
                predict (log) probabilities

        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks

        skipConnect: (np.array, dim = 1)
                    skip connections

        Return
        ------
        beta: (np.array, dim = (output len, out channel))
                backward probabilities

        """
        
        S, T = len(extSymbols), len(logits)
        beta = np.zeros(shape=(T, S))
        self.logits = logits
       # print(extSymbols)


        # -------------------------------------------->

        # Your Code goes here
        beta[T-1, S-1] = 1
        beta[T-1, S-2] = 1
        beta[T-1, 0:S-2] = 0
        for t in range(T-2,-1,-1):
            beta[t,S-1] = beta[t+1,S-1] * logits[t+1, extSymbols[S-1]]
            for i in range(S-2,-1,-1):
                beta[t,i] = beta[t+1, i] * logits[t+1, extSymbols[i]] + beta[t+1,i+1] * logits[t+1, extSymbols[i+1]]
                if i < S -2 and skipConnect[i+2]:
                    beta[t,i] += beta[t+1,i+2] * logits[t+1, extSymbols[i+2]]
        # <---------------------------------------------

        return beta

    def postProb(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array)
                forward probability

        beta: (np.array)
                backward probability

        Return
        ------
        gamma: (np.array)
                posterior probability

        """
        T = beta.shape[0]
        S = beta.shape[1]

        gamma = np.zeros(shape = (T,S))
        sumgamma = [0] *T
       

        # -------------------------------------------->

        # Your Code goes here
        for t in range(T):
            sumgamma[t] = 0
            for i in range(S):
                gamma[t,i] = alpha[t,i] * beta[t,i]
                sumgamma[t] += gamma[t,i]
            for i in range(S):
                gamma[t,i] = gamma[t,i] / sumgamma[t]
        
        # <---------------------------------------------

        return gamma
