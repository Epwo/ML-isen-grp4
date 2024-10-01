import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define project paths
PROJECT_PATH = Path(__file__).parents[1]
sys.path.append(PROJECT_PATH)
SRC_PATH = Path(__file__).resolve().parents[0]
sys.path.append(str(SRC_PATH))

from sklearn.linear_model import Lasso, Ridge

# imports customs functions
from ML.regLasso.main import LassoRegressionCustom
from ML.regRidge.main import RidgeRegressionCustom
# import customs pretreatment functions
from Pretreatment.ModelTrainer import ModelTrainer

modellList = [
    {"model": LassoRegressionCustom, "params": {"alpha": 0.1}},
    {"model": RidgeRegressionCustom, "params": {"alpha": 0.1}},
    {"model": Lasso, "params": {"alpha": 0.01}},
    {"model": Ridge, "params": {"alpha": 0.01}},
]

class Runner:
    def __init__(self):
        self.modeltrainer = ModelTrainer()

    def run(self):
        x_train, y_train, x_test, y_true = self.modeltrainer.process_data()

        for model_info in modellList:
            model_class = model_info["model"]
            params = model_info.get("params", {})
            
            model = model_class(**params)
            
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            print(f"R2 score for {model_class.__name__}: {r2_score(y_true=y_true, y_pred=y_pred)}")


if __name__ == "__main__":
    runner = Runner()
    runner.run()
