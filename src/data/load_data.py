import yaml
import pandas as pd
import argparse


def read_parameter(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
        return config


def get_data(config_path):
    config = read_parameter(config_path)
    data_path = config["data_source"]["source"]
    df = pd.read_csv(data_path)
    return df


def load_and_save(config_path):
    config = read_parameter(config_path)
    df = get_data(config_path)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path, index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml", )
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)
