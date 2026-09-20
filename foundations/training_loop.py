import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        w,b = np.zeros(X.shape[1]),0
        for epoch_idx in range(epochs):
            y_pred = self.get_res(X,w,b)
            grad = self.get_grad(X,y,y_pred)
            w,b = w - lr * grad[0], b - lr * grad[1]
        return (np.round(w,5), round(b,5))
        
    def get_grad(self,X,y,y_pred):  return (2/X.shape[0] * np.matmul(X.T,(y_pred-y))),(2/X.shape[0] * np.sum(y_pred-y))

    def get_res(self,X,w,b): return np.matmul(X,w)+b
