import yaml
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split


def read_parameter(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
        return config


def split_and_save_data(config_path):
    config = read_parameter(config_path)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    train_path = config["split_data"]["train_path"]
    test_path = config["split_data"]["test_path"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]
    df = pd.read_csv(raw_data_path)
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_save_data(config_path=parsed_args.config)
