import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        eps = 1e-5
        mu = (1/x.shape[0]) * np.sum(x,axis=0)
        var = (1/x.shape[0]) * np.sum(np.square(x-mu),axis=0)
        x_hat = ((x-mu)/np.sqrt(var+eps))
        return np.round(gamma * x_hat + beta,5)