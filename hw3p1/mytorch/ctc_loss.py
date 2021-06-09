import numpy as np
from ctc import *
import math

class CTCLoss(object):
    """CTC Loss class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument:
                blank (int, optional) – blank label index. Default 0.
        """
        # -------------------------------------------->
        # Don't Need Modify
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # -------------------------------------------->
        # Don't Need Modify
        return self.forward(logits, target, input_lengths, target_lengths)
        # <---------------------------------------------

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward.

        Computes the CTC Loss.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        loss: scalar
            (avg) divergence between the posterior probability γ(t,r) and the input symbols (y_t^r)

        """
        # -------------------------------------------->
        # Don't Need Modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        # <---------------------------------------------

        #####  Attention:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # -------------------------------------------->
        # Don't Need Modify
        B, _ = target.shape
        totalLoss = np.zeros(B)

        # <---------------------------------------------
    
       

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Extend Sequence with blank ->
            #     Compute forward probabilities ->
            #     Compute backward probabilities ->
            #     Compute posteriors using total probability function
            #     Compute Expected Divergence and take average on batches
            # <---------------------------------------------
            

            # -------------------------------------------->

            # Your Code goes here
            ctcobj = CTC(self.BLANK)
            extSymbols, skipConnect= ctcobj.targetWithBlank(self.target[b,:self.target_lengths[b]])
      
            
            alpha = ctcobj.forwardProb(logits[:self.input_lengths[b],b], extSymbols, skipConnect)  
            beta = ctcobj.backwardProb(logits[:self.input_lengths[b],b], extSymbols, skipConnect)
            gamma = ctcobj.postProb(alpha,beta)
            
            true = logits[:self.input_lengths[b],b,extSymbols]
            div = 0
            
            div += np.sum(-gamma * np.log(true))
               
            totalLoss[b] = np.mean(div)
            
            # <---------------------------------------------

        return np.mean(totalLoss)

    def backward(self):
        """CTC loss backard.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        dY: scalar
            derivative of divergence wrt the input symbols at each time.

        """
        # -------------------------------------------->
        # Don't Need Modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        # <---------------------------------------------
    

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # <---------------------------------------------
            
            ctcobj = CTC(self.BLANK)
            extSymbols, skipConnect= ctcobj.targetWithBlank(self.target[b,:self.target_lengths[b]])
      
            
            alpha = ctcobj.forwardProb(self.logits[:self.input_lengths[b],b], extSymbols, skipConnect)  
            beta = ctcobj.backwardProb(self.logits[:self.input_lengths[b],b], extSymbols, skipConnect)
            gamma = ctcobj.postProb(alpha,beta)
         
        
            true = self.logits[:self.input_lengths[b],b,extSymbols]
            
            
            for s in range(self.input_lengths[b]):  
                finished = []
               # print('s',s)
                for pos, ext in enumerate(extSymbols):
                    
                    to_calculate = [i for i in range(len(extSymbols)) if ext == extSymbols[i]]
               
                    if pos not in finished:
                        finished.extend(to_calculate)
               
                        backsum = np.sum([-gamma[s,p]/true[s,p] for p in to_calculate])
                 
                        for e in extSymbols:
                            if e == ext:
                              dY[s,b,e] = backsum

         
        return dY
