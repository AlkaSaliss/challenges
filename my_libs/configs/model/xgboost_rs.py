from sklearn.model_selection import ParameterSampler
import numpy as np 


model_name = "xgboost"
params_grid = {
    "booster": ["gbtree", "gblinear"],
    "learning_rate": np.arange(0.01, 2.0, 0.01),
    "max_depth": range(5, 50, 5),
    "tree_method": ["auto"],
    "reg_lambda": np.arange(0.01, 2.0, 0.01), 
    "reg_alpha": np.arange(0.01, 2.0, 0.01),
    "n_estimators": [500, 1000, 2000],
    "n_jobs": [-1],
    "random_state": [123],
    "verbose": [2]
}

model_args = list(ParameterSampler(params_grid, n_iter=200, random_state=123))

out_path = "/home/alka/Documents/zindi_challenges/malawi_flood_prediction/random_search_results"
exp_name = "xgboost_1"
