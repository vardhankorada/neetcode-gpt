import numpy as np
from typing import Tuple, List


class Solution:

    def batch_norm(
        self,
        x: List[List[float]],
        gamma: List[float],
        beta: List[float],
        running_mean: List[float],
        running_var: List[float],
        momentum: float,
        eps: float,
        training: bool,
    ) -> Tuple[List[List[float]], List[float], List[float]]:

        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        if training:
            mu_b = np.mean(x, axis=0)
            var_b = np.var(x, axis=0)

            x_hat = (x - mu_b) / np.sqrt(var_b + eps)

            running_mean = (1 - momentum) * running_mean + momentum * mu_b
            running_var = (1 - momentum) * running_var + momentum * var_b

            y = gamma * x_hat + beta
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
            y = gamma * x_hat + beta

        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist(),
        )