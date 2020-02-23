from utils import cross_validate_sk_regressor, random_search_cv
import json
import path
import argparse
import time
import os
import pandas as pd



def main(config):

    start_t = time.time()
    # get data params
    if config["data_path"].endswith(".csv") or config["data_path"].endswith(".txt"):
        data = pd.read_csv(config["data_path"])
    else:
        data = pd.read_feather(config["data_path"])
    
    n_splits = config.get("n_splits", 5)
    shuffle = config.get("shuffle", False)
    features_cols = config["features_cols"]
    target_col = config["target_col"]

    # get model params
    model_name = config["model_name"]
    model_args = config["model_args"]
    train_args = config.get("train_args")
    
    out_path = config.get("out_path", "./")
    exp_name = config.get("exp_name", f"{model_name}") + f"_{str(int(1000*time.time()))}"
    out_file = os.path.join(out_path, exp_name+".json")

    exp_result = random_search_cv(data, n_splits, shuffle, model_name, model_args, train_args, features_cols, target_col)

    duration = time.time() - start_t

    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)

    exp_result["duration"] = f"{h}h : {m}m : {s:.1f}s"
    exp_result["model_config"] = config["model_config"]
    exp_result["data_config"] = config["data_config"]

    with open(out_file, "w") as f:
        json.dump(exp_result, f)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="random hyperparameters search with sklearn-API-like regressors (SVR, random forest, xgboost, lightgbm, ...)")
    parser.add_argument("--model_config", type=str, help="path to the python script containing model parameters", required=True)
    parser.add_argument("--data_config", type=str, help="path to the json file containing data configuration", required=True)
    args = parser.parse_args()

    config = {}
    with open(args.model_config) as f1, open(args.data_config) as f2:
        exec(f1.read(), config)
        config.update(json.load(f2))
    
    config["model_config"] = os.path.abspath(args.model_config) 
    config["data_config"] = os.path.abspath(args.data_config)

    main(config)