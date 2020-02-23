import pandas as pd 
import numpy as np 
from scipy.special import erfinv
from sklearn.metrics import mean_squared_error
from data import KfoldSplitter
from models import RegressorModel
import multiprocessing
import tqdm
import json


def to_gauss(x): return np.sqrt(2) * erfinv(x)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def gauss_rank_one(x):
    # based on : https://www.kaggle.com/benjibb/denoised-autoencoder-for-feature-engineering/
    sorted_ids = np.argsort(x)
    uniform = np.linspace(start=-0.99, stop=0.99, num=len(x))
    normal = to_gauss(uniform)

    return normal[np.argsort(sorted_ids)]


def swap_noise(x, p):
    np.random.seed(123)
    assert 0 < p <= 1, f"`p` parameter must be between 0 and 1, but {p} provided"

    n_rows, n_cols = x.shape

    n_select = round(p*n_rows)

    # for each column
    for i in range(n_cols):
        # select randomly a fraction p of rows to swap values for
        random_ids = np.random.choice(n_rows, size=n_select, replace=False)
        # select randomly candidate values to use for replacement from the same column
        random_values = np.random.choice(x[:, i], size=n_select)

        assert len(random_ids) == len(random_values), f"the permutation indices and replacement values don't have the same size : {len(random_ids)} and {len(random_values)}" 

        # replace the values
        x[random_ids, i] = random_values
    
    return x


class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def cross_validate_sk_regressor(data, n_splits, shuffle, model_name, model_args, train_args, features_cols, target_col):
    # Create data splitter 
    data_spliter = KfoldSplitter(data, n_splits, shuffle)

    result = []

    for train_data, test_data in data_spliter:
        # prepare data
        if model_name == "catboost":
            X_train, y_train = train_data[features_cols], train_data[target_col]
            X_test, y_test = test_data[features_cols], test_data[target_col]
            #print(type(X_train), type(X_test))
        else:
            X_train, y_train = train_data[features_cols].values, train_data[target_col].values
            X_test, y_test = test_data[features_cols].values, test_data[target_col].values
        
        try:
            # create and fit model
            model = RegressorModel(model_name, **model_args)
            if train_args:
                model.fit(X_train, y_train, **train_args)
            else:
                model.fit(X_train, y_train)
            
            # evaluate model
            result.append(float(model.evaluate(X_test, y_test)))

        except:
            raise
            
        
        
        
    
    return {"param__" + "__".join([f"{k}:{v}" for k, v in model_args.items()]) : result}
    

def random_search_cv(data, n_splits, shuffle, model_name, model_args, train_args, features_cols, target_col):
    list_args = [(data, n_splits, shuffle, model_name, arg, train_args, features_cols, target_col) for arg in model_args]

    with MyPool() as pool:
        results = pool.starmap(cross_validate_sk_regressor, tqdm.tqdm(list_args))
    
    results = {k: v for res in results for k, v in res.items()}

    return results



def train_single(model_name, model_args, X, y, train_args):
    model = RegressorModel(model_name, **model_args)
    if train_args:
        model.fit(X, y, **train_args)
    else:
        model.fit(X, y)
    return model

def retrain_cv(path_to_xp):

    with open(path_to_xp) as f:
        xp_result = json.load(f)

    # get best params 
    _, best_params = load_xp_results(path_to_xp)

    # get data config
    with open(xp_result["data_config"]) as f:
        data_config = json.load(f)
    
    if data_config["data_path"].endswith(".csv") or data_config["data_path"].endswith(".txt"):
        data = pd.read_csv(data_config["data_path"])
    else:
        data = pd.read_feather(data_config["data_path"])
    n_splits = data_config.get("n_splits", 5)
    shuffle = data_config.get("shuffle", False)
    features_cols = data_config["features_cols"]
    target_col = data_config["target_col"]

    data_spliter = KfoldSplitter(data, n_splits, shuffle)

    # get model name
    model_config = {}
    with open(xp_result["model_config"]) as f:
        exec(f.read(), model_config)
    
    model_name = model_config["model_name"]
    train_args = model_config.get("train_args")

    list_args = []
    for train_data, test_data in data_spliter:
        # prepare data
        X_train, y_train = train_data[features_cols].values, train_data[target_col].values
        list_args.append(
            (model_name, best_params, X_train, y_train, train_args)
        )
    
    # train k_fold
    with MyPool() as pool:
        list_models = pool.starmap(train_single, tqdm.tqdm(list_args))
    
    return list_models


def load_xp_results(path_to_xp):
    with open(path_to_xp) as f:
        xp_results = json.load(f)
    
    list_params = []
    list_scores = []
    list_scores_std = []
    for p in xp_results:
        if p.startswith("param__"):
            list_params.append(p)
            list_scores.append(np.mean(xp_results[p]))
            list_scores_std.append(np.std(xp_results[p]))
    
    result = pd.DataFrame({"parameters": list_params, "mean_scores": list_scores, "std_scores": list_scores_std})
    best_params = {}
    for item in result.sort_values("mean_scores").iloc[0]["parameters"].split("__")[1:]:
        k, v = item.split(":")[0], item.split(":")[1]
        try:
            v = eval(v)
        except NameError:
            v = v
        best_params[k] = v
    # best_params = {item.split(":")[0] :eval(item.split(":")[1])
    #                 for item in result.sort_values("mean_scores").iloc[0]["parameters"].split("__")[1:] }

    return result, best_params

def predict_cv(list_models, X):
    list_preds = []
    for model in list_models:
        list_preds.append(np.expand_dims(model.predict(X), 1))
    
    return np.mean(np.hstack(list_preds), axis=1)
    
