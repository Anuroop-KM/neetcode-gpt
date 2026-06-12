import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        pass
        x_arr=np.array(x,dtype=np.float64)
        gamma_arr = np.array(gamma, dtype=np.float64)
        beta_arr = np.array(beta, dtype=np.float64)
        running_mean_arr = np.array(running_mean, dtype=np.float64)
        running_var_arr = np.array(running_var, dtype=np.float64)

        if training:
            batch_mean=np.mean(x_arr,axis=0)
            batch_var=np.var(x_arr,axis=0)

            x_hat = (x_arr - batch_mean) / np.sqrt(batch_var + eps)
            running_mean_arr = (1 - momentum) * running_mean_arr + momentum * batch_mean
            running_var_arr = (1 - momentum) * running_var_arr + momentum * batch_var
        else:
            x_hat = (x - running_mean_arr) / np.sqrt(running_var_arr + eps)

        out = gamma * x_hat + beta
        return (np.round(out, 4).tolist(), np.round(running_mean_arr, 4).tolist(), np.round(running_var_arr, 4).tolist())   
            