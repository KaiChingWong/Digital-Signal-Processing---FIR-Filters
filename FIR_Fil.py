""" 1 ECG Filtering: Task 2 """

import numpy as np

class FIR_filter:
    def __init__(self,_coefficients):
        self.M = len(_coefficients) # M is number of taps
        self.h = _coefficients # h is number of coefficients
        self.buffer = np.zeros(self.M)
        
    def filter(self,v):
        self.buffer = np.delete(self.buffer,len(self.buffer)-1)
        self.buffer = np.insert(self.buffer,0,v)
        result = 0
        for i in range (self.M):
            result = result + self.buffer[i]*self.h[i]
        return result
    def reset(self):
        self.buffer = np.zeros(self.M)