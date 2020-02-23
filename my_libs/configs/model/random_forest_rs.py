from sklearn.model_selection import ParameterSampler
import numpy as np 



model_name = "random_forest"
params_grid = {
    "n_estimators": range(10, 1050, 50),
    "max_depth": range(2, 50, 3),
    "min_samples_leaf": range(1, 11),
    "max_features": ["auto", "sqrt", "log2"],
    "n_jobs": [-1],
    "random_state": [123]
}

model_args = list(ParameterSampler(params_grid, n_iter=100, random_state=123))

out_path = "/home/alka/Documents/zindi_challenges/malawi_flood_prediction/random_search_results"
exp_name = "random_forest_1"
