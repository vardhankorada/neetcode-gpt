import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        ans = np.matmul(X,weights)
        return np.round(ans,5)

    def get_error(self, model_pred: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        return np.round(np.mean(np.square(model_pred-ground_truth)),5)
