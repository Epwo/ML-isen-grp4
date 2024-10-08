import sys
from pathlib import Path

import pandas as pd
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define project paths
PROJECT_PATH = Path(__file__).parents[1]
sys.path.append(PROJECT_PATH)
SRC_PATH = Path(__file__).resolve().parents[0]
sys.path.append(str(SRC_PATH))

# import sklearn functions
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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
    # {"model":SVC, "params": {"kernel":"linear","random_state":42}, "type": "class"},
    # {"model": DecisionTreeClassifier, "params": {}, "type": "class"}

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
            # start time
            startT = time.time()

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            # calculate metrics
            elapsed = time.time() - startT

            nameModel = model_class.__name__
            if model_info["type"] == "regr":
                # Store results
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                print(f"{nameModel} - time {elapsed}")
                results.append({
                    "Model": nameModel,
                    "MAE": mae,
                    "MSE": mse,
                    "R2 Score": r2,
                    "time": elapsed
                })
            elif model_info["type"] == "class":
                acc = accuracy_score(y_true, y_pred)
                results.append({
                    "Model": nameModel,
                    "accuracy": acc,
                    "time": elapsed
                })

                # Select only numeric features for the correlation matrix
                x_train_trans = self.model_trainer.dataframe.drop(columns=[self.model_trainer.target])
                numeric_x_train = x_train_trans.select_dtypes(include=["float64", "int64"])
                correlation_matrix = numeric_x_train.corr()

                # Plotting the correlation matrix using seaborn
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
                plt.title('Correlation Matrix of Training Features')
                plt.savefig(f"corrMat-{nameModel}.png")
                print("Correlation matrix exported to 'corrMat.png'.")

        # Create a DataFrame from the results and export to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv("model_comparison_results.csv", index=False)
        print("Model comparison results exported to 'model_comparison_results.csv'.")


if __name__ == "__main__":
    runner = Runner()
    runner.run()
