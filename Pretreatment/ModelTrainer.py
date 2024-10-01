import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_PATH = Path(__file__).parents[1]
sys.path.append(PROJECT_PATH)

SRC_PATH = Path(__file__).resolve().parents[0]
sys.path.append(str(SRC_PATH))

from ConfigLoader import ConfigLoader


class ModelTrainer(ConfigLoader):
    def __init__(self):
        super().__init__()
        self.dataframe = self.get_dataframe()
        self.target = self.config["target"]

    def get_dataframe(self):
        file_path = os.path.join(PROJECT_PATH,'data',self.config["files"])
        delimiter = self.config["delimiter"]
        return pd.read_csv(file_path,delimiter=delimiter)

    def get_train_test_split(self, df):
        """
        Split the dataframe into training and testing datasets.
        """
        split_index = int(0.8 * len(df))
        df_train = df.iloc[:split_index]
        df_test = df.iloc[split_index:]

        return df_train, df_test

    def prepare_data(self, data, is_train=True):
        """
        Prepare the training or testing data by selecting the features and the target variable.
        """
        x_data = data.drop(columns=[self.target])
        if is_train:
            y_data = data[self.target]
            return x_data, y_data
        return x_data


    def process_data(self):
        """
        Process the data and train the model, returning the scaled training and testing data.
        """
        self.get_dataframe()

        data_train, data_test = self.get_train_test_split(self.dataframe.dropna())

        x_train, y_train = self.prepare_data(data_train, is_train=True)
        x_test = self.prepare_data(data_test, is_train=False)
        y_true = data_test[self.target]

        print(f"y_names: {self.target}")

        print(f"X_train shape = {x_train.shape}")
        print(f"X_test shape = {x_test.shape}")
        print(f"Y_train shape = {y_train.shape}")
        print(f"Y_true shape = {y_true.shape}")

        return x_train.values, y_train.values, x_test.values, y_true.values
