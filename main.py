import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# Define project paths
PROJECT_PATH = Path(__file__).parents[1]
sys.path.append(PROJECT_PATH)
SRC_PATH = Path(__file__).resolve().parents[0]
sys.path.append(str(SRC_PATH))

from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVC
# imports customs functions
from ML.regLasso.main import LassoRegressionCustom
from ML.regRidge.main import RidgeRegressionCustom
# import customs pretreatment functions
from Pretreatment.ModelTrainer import ModelTrainer

model_list = [
    {"model": LassoRegressionCustom, "params": {"alpha": 0.01}, "type": "regr"},
    {"model": RidgeRegressionCustom, "params": {"alpha": 0.01}, "type": "regr"},
    {"model": Lasso, "params": {"alpha": 0.01}, "type": "regr"},
    {"model": Ridge, "params": {"alpha": 0.01}, "type": "regr"},
    {"model":SVC, "params": {"kernel":"linear","random_state":42}, "type": "class"}

]

class Runner:
    def __init__(self):
        self.model_trainer = ModelTrainer()

    def run(self):
        x_train, y_train, x_test, y_true = self.model_trainer.process_data()
        results = []

        for model_info in model_list:

            model_class = model_info["model"]
            params = model_info.get("params", {})
            
            model = model_class(**params)
            
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            #calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            # Store results
            results.append({
                "Model": model_class.__name__,
                "MAE": mae,
                "MSE": mse,
                "R2 Score": r2
            })
            if model_info["type"] == "class":
                acc = accuracy_score(y_true, y_pred)
                results[-1]["Accuracy"] = acc


        # Create a DataFrame from the results and export to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv("model_comparison_results.csv", index=False)
        print("Model comparison results exported to 'model_comparison_results.csv'.")

if __name__ == "__main__":
    runner = Runner()
    runner.run()
