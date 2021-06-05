import yaml
import pandas as pd
import argparse
import json
from sklearn.ensemble import ExtraTreesRegressor


def read_parameter(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
        return config


def get_data(config_path):
    config = read_parameter(config_path)
    data_path = config["data_source"]["source"]
    df = pd.read_csv(data_path)
    return df


def get_feature_importance(config_path):
    config = read_parameter(config_path)
    target = [config["base"]["target_col"]]
    feature_importance_file = config["reports"]["feature_importance"]
    df = get_data(config_path)
    df = df.dropna()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    fi = ExtraTreesRegressor()
    fi.fit(X, y)
    fi_arr = fi.feature_importances_
    with open(feature_importance_file, "w") as f:
        importance = {
            "T": fi_arr[0],
            "TM": fi_arr[1],
            "Tm": fi_arr[2],
            "SLP": fi_arr[3],
            "H": fi_arr[4],
            "VV": fi_arr[5],
            "V": fi_arr[6],
            "VM": fi_arr[7]
        }
        json.dump(importance, f, indent=4)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    get_feature_importance(config_path=parsed_args.config)
