from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.metrics import mean_squared_error
import numpy as np 


dict_models = {
    "svm": SVR,
    "lasso": Lasso,
    "random_forest": RandomForestRegressor,
    "light_gbm": LGBMRegressor,
    "catboost": CatBoostRegressor,
    "xgboost": XGBRegressor
}


class RegressorModel(BaseEstimator):

    def __init__(self, model_name, **kwargs):
        assert model_name in dict_models.keys(), f"Model name should be selected from : {', '.join(dict_models.keys())}"
        self.model_name = model_name
        self.model = dict_models[model_name](**kwargs)
    
    def fit(self, X, y=None, **kwargs):
        if self.model_name != "catboost":
            X, y = check_X_y(X, y)
        if kwargs:
            self.model.fit(X, y, **kwargs)
        else:
            self.model.fit(X, y)
        return self
    
    def predict(self, X):
        check_is_fitted(self.model)
        if self.model_name != "catboost":
            X = check_array(X)
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)
    
    def set_params(self, **parameters):
        self.model.set_params(**parameters)
        return self
    
    def evaluate(self, X, y):
        if self.model_name != "catboost":
            check_X_y(X, y)

        preds = self.predict(X)
        eval_metric = np.sqrt(mean_squared_error(y, preds))
        return eval_metric
