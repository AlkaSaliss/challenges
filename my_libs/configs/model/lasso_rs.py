from sklearn.model_selection import ParameterSampler


model_name = "lasso"
params_grid = {
    "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 7.0, 10.0],
    "fit_intercept": [True, False],
    "normalize": [True, False],
    "random_state": [123],
    "max_iter": [10000]
}

model_args = list(ParameterSampler(params_grid, n_iter=100, random_state=123))

out_path = "/home/alka/Documents/zindi_challenges/malawi_flood_prediction/random_search_results"
exp_name = "lasso_1"
