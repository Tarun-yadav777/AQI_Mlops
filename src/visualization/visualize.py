import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import yaml
import json
import joblib
import argparse


def read_parameter(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
        return config


def correlation_plot(config_path):
    config = read_parameter(config_path)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    figure_path = config["reports"]["plots"]
    df = pd.read_csv(raw_data_path)
    sns_plot = sns.heatmap(df.corr(), annot=True, cmap='viridis')
    sns_plot.figure.savefig(os.path.join(figure_path, "correlation_plot.png"))


def feature_importance_plot(config_path):
    config = read_parameter(config_path)
    feature_importance = config["reports"]["feature_importance"]
    figure_path = config["reports"]["plots"]
    with open(feature_importance) as f:
        data = json.load(f)
        df = pd.Series(data)
        df.plot(kind='barh', color='purple').figure.savefig(os.path.join(figure_path, "feature_importance_plot.png"))


def predict_value_plot(config_path):
    config = read_parameter(config_path)
    model_dir = config["model_dir"]
    target = config["base"]["target_col"]
    figure_path = config["reports"]["plots"]
    model_path = os.path.join(model_dir, "model.joblib")
    test_data_path = config["split_data"]["test_path"]
    test = pd.read_csv(test_data_path, sep=",")
    test_y = test[target]
    test_x = test.drop(target, axis=1)
    model = joblib.load(model_path)
    pred_y = model.predict(test_x)
    sns_plot = sns.scatterplot(test_y, pred_y)
    sns_plot.figure.savefig(os.path.join(figure_path, "predict_scatter_plot.png"))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    correlation_plot(config_path=parsed_args.config)
    feature_importance_plot(config_path=parsed_args.config)
    predict_value_plot(config_path=parsed_args.config)
