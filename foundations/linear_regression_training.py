import numpy as np
from numpy.typing import NDArray
class Solution:
    learning_rate = 0.01
    def get_derivative(self, model_prediction, ground_truth, N, X, desired_weight):
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N
    def get_model_prediction(self, X, weights):
        return np.squeeze(np.matmul(X, weights))
    def train_model(self,X,Y,num_iterations,initial_weights):
        weights = initial_weights
        for it in range(num_iterations):
            preds = self.get_model_prediction(X,weights)
            for i in range(len(initial_weights)):
                gradient = self.get_derivative(preds,Y,Y.shape[0],X,i)
                weights[i] = np.subtract(weights[i],self.learning_rate*gradient)
        return np.round(weights,5)
