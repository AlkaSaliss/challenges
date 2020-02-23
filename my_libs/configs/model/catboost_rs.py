from sklearn.model_selection import ParameterSampler
import numpy as np 


model_name = "catboost"
params_grid = {
    "iterations": range(10, 2000, 50),
    # "max_leaves": range(10, 100, 5),
    "depth": range(2, 17),
    "learning_rate": np.arange(0.01, 2.0, 0.01),
    # "min_data_in_leaf": range(5, 50, 5),
    # "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1. ],
    "rsm": [0.5, 0.6, 0.7, 0.8, 0.9, 1. ],
    "l2_leaf_reg": np.arange(0.0, 5.0, 0.1),
    # "task_type": ["GPU"],
    "random_state": [123]
}

model_args = list(ParameterSampler(params_grid, n_iter=200, random_state=123))

train_args = {
    "cat_features": [43]
}

out_path = "/home/alka/Documents/zindi_challenges/malawi_flood_prediction/random_search_results"
exp_name = "catboost_1"
