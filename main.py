import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define project paths
PROJECT_PATH = Path(__file__).parents[1]
sys.path.append(PROJECT_PATH)
SRC_PATH = Path(__file__).resolve().parents[0]
sys.path.append(str(SRC_PATH))

from ML.regLasso.main import LassoRegression
from Pretreatment.ModelTrainer import ModelTrainer

modellList = [LassoRegression]


class Runner:
    def __init__(self):
        self.modeltrainer = ModelTrainer()

    def run(self):
        x_train, y_train, x_test, y_true = self.modeltrainer.process_data()

        for model in modellList:
            model = model()
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            print(r2_score(y_pred=y_pred,y_true=y_true))


if __name__ == "__main__":
    runner = Runner()
    runner.run()
