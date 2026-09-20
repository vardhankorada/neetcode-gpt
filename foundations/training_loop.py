import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        w,b = np.zeros(X.shape[1]),0
        for epoch_idx in range(epochs):
            y_pred = self.get_res(X,w,b)
            loss = self.get_loss(y,y_pred)
            grad = self.get_grad(X,y,y_pred)
            w,b = w - lr * grad[0], b - lr * grad[1]
        return (np.round(w,5), round(b,5))
        
    def get_grad(self,X,y,y_pred): 
        n = X.shape[0]
        return (2/n * np.matmul(X.T,(y_pred-y))),(2/n * np.sum(y_pred-y))

    def get_res(self,X,w,b): return np.matmul(X,w)+b

    def get_loss(self,y,y_pred): return np.mean(np.pow(y_pred-y,2))

