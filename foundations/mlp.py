import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        h = x
        for ind in range(0,len(weights)):
            z = self.linear(weights[ind],biases[ind],h)
            if ind < len(weights)-1: h = self.relu(z)
        return np.round(z,5)

    def relu(self,z): return np.maximum(0,z)

    def linear(self,W,b,h): return h @ W + b 
