import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        rms_x = np.sqrt(np.mean(np.pow(x,2))+eps)
        x_hat = x/rms_x
        return np.round(gamma * x_hat,4)
