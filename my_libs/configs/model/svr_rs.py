from sklearn.model_selection import ParameterSampler
import numpy as np 



model_name = "svm"
params_grid = {
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "degree": range(10),
    "gamma": np.arange(0.005, 0.5, 0.01),
    "C": np.arange(0.1, 5.1, 0.1),
    "epsilon": np.arange(0.05, 0.5, 0.01)
}

model_args = list(ParameterSampler(params_grid, n_iter=100, random_state=123))

out_path = "/home/alka/Documents/zindi_challenges/malawi_flood_prediction/random_search_results"
exp_name = "svm_1"
