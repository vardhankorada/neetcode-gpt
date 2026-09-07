import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        h = x
        for ind in range(0,len(weights)):
            z = self.linear(weights[ind],biases[ind],h)
            if ind < len(weights)-1: h = self.relu(z)
        return z

    def relu(self,z): return np.maximum(0,z)

    def linear(self,W,b,h): return h @ W + b 
