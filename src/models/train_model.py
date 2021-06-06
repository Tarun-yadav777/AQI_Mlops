import pandas as pd
import yaml
import os
import json
import argparse
from sklearn.ensemble import RandomForestRegressor
import joblib


def read_parameter(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
        return config


def train_model(config_path):
    config = read_parameter(config_path)
    train_data_path = config["split_data"]["train_path"]
    models_dir = config["model_dir"]
    target = config["base"]["target_col"]
    report_params = config["reports"]["params"]

    n_estimator = config["estimators"]["RandomForestRegressor"]["params"]["n_estimators"]
    min_samples_split = config["estimators"]["RandomForestRegressor"]["params"]["min_samples_split"]
    min_samples_leaf = config["estimators"]["RandomForestRegressor"]["params"]["min_samples_leaf"]
    max_features = config["estimators"]["RandomForestRegressor"]["params"]["max_features"]
    max_depth = config["estimators"]["RandomForestRegressor"]["params"]["max_depth"]

    with open(report_params, "w") as f:
        scores = {
            "n_estimators": n_estimator,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "max_depth": max_depth
        }
        json.dump(scores, f, indent=4)

    train = pd.read_csv(train_data_path, sep=",")

    train_y = train[target]

    train_x = train.drop(target, axis=1)

    model = RandomForestRegressor(n_estimators=n_estimator, min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf, max_features=max_features, max_depth=max_depth)
    model.fit(train_x, train_y)

    model_path = os.path.join(models_dir, "model.joblib")
    joblib.dump(model, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_model(config_path=parsed_args.config)
