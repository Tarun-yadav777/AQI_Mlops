import numpy as np
import pandas as pd
import yaml
import os
import json
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


def read_parameter(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
        return config


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def eval_model(config_path):
    config = read_parameter(config_path)
    model_dir = config["model_dir"]
    target = config["base"]["target_col"]
    model_path = os.path.join(model_dir, "model.joblib")
    test_data_path = config["split_data"]["test_path"]
    report_score = config["reports"]["scores"]
    test = pd.read_csv(test_data_path, sep=",")
    test_y = test[target]
    test_x = test.drop(target, axis=1)
    model = joblib.load(model_path)
    pred_y = model.predict(test_x)
    mae = mean_squared_error(test_y, pred_y)
    mse = mean_squared_error(test_y, pred_y)
    r2 = r2_score(test_y, pred_y)
    with open(report_score, "w") as f:
        scores = {
            "mae": mae,
            "mse": mse,
            "r2": r2
        }
        json.dump(scores, f, indent=4)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    eval_model(config_path=parsed_args.config)
