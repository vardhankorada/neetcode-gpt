import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,x: List[float],W1: List[List[float]], b1: List[float],W2: List[List[float]], b2: List[float],y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        ans = {}
        W1,W2,b1,b2,x,y_true = np.array(W1),np.array(W2),np.array(b1),np.array(b2),np.array(x),np.array(y_true)
        z1 = W1 @ x + b1
        a1 = np.where(z1>0,z1,0)
        z2 = W2 @ a1 + b2
        loss = np.mean(np.power(z2-y_true,2))
        
        # Calculate gradients
        db2 = 2/y_true.shape[0] * (z2-y_true)
        dW2 = np.outer(2/y_true.shape[0] * (z2-y_true),a1)
        db1 = np.multiply((2/(y_true.shape[0]) * (z2-y_true)) @ W2,(z1>0).astype(float))
        dW1 = np.outer(np.multiply((2/(y_true.shape[0]) * (z2-y_true)) @ W2,(z1>0).astype(float)),x)
        
        # Round and save answers to 4 decimal places
        ans['loss'] = round(loss.item(), 4)
        ans['db2'] = np.round(db2, 4).tolist()
        ans['dW2'] = np.round(dW2, 4).tolist()
        ans['db1'] = np.round(db1, 4).tolist()
        ans['dW1'] = np.round(dW1, 4).tolist()
        return ans

