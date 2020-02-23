from sklearn.model_selection import ParameterSampler
import numpy as np 


model_name = "light_gbm"
params_grid = {
    "boosting_type": ['gbdt', 'dart', 'goss'],
    "num_leaves": range(10, 100, 5),
    "max_depth": range(2, 50, 3),
    "learning_rate": np.arange(0.05, 2.0, 0.01),
    "n_estimators": range(10, 1050, 50),
    "min_child_samples": range(5, 50, 5),
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1. ],
    "subsample_freq": [1, 5, 10],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1. ],
    "reg_alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ],
    "reg_lambda": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ],
    "random_state": [123]
}

model_args = list(ParameterSampler(params_grid, n_iter=200, random_state=123))

train_args = {
    "categorical_feature": [43]
}

out_path = "/home/alka/Documents/zindi_challenges/malawi_flood_prediction/random_search_results"
exp_name = "lgbm_1"
