import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_pred_clip = np.clip(y_pred, 1e-7 , 1.0 - 1e-7)
        r=(y_true*np.log(y_pred_clip)) + ((1-y_true)*(np.log(1-y_pred_clip)))
        return float(round(-np.mean(r),4))
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        y_pred_clip = np.clip(y_pred, 1e-7 , 1.0 - 1e-7)
        loss_per_sample =np.sum(y_true * np.log(y_pred_clip), axis=1)
        return float(round(-np.mean(loss_per_sample),4))
        pass

    
