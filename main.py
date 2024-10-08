import itertools  # Importer itertools pour les combinaisons de paramètres
import sys
from pathlib import Path

import pandas as pd
<<<<<<< HEAD
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import (MinMaxScaler, Normalizer,
                                   QuantileTransformer, StandardScaler)
from sklearn.tree import DecisionTreeClassifier
=======
import time
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
>>>>>>> b695b5fa35bbe2a7a3e07fe7a236b9257c420d9b

# Define project paths
PROJECT_PATH = Path(__file__).parents[1]
sys.path.append(str(PROJECT_PATH))
SRC_PATH = Path(__file__).resolve().parents[0]
sys.path.append(str(SRC_PATH))

<<<<<<< HEAD
# Imports custom functions
from ML.arbres.main import DecisionTree
from Pretreatment.ModelTrainer import ModelTrainer

model_list = [
    {"model": DecisionTreeClassifier, "params": {"max_depth": [1, 2, 3, 4, 5], "criterion": ["gini", "entropy"]}, "type": "class"},
    {"model": DecisionTree, "params": {"max_depth": [1, 2, 3, 4, 5]}, "type": "class"}
=======
# import sklearn functions
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# imports customs functions
from ML.regLasso.main import LassoRegressionCustom
from ML.regRidge.main import RidgeRegressionCustom
from ML.SVM.supportvectormachine import SupportVectorMachineCustom
# import customs pretreatment functions
from Pretreatment.ModelTrainer import ModelTrainer

model_list = [
    #{"model": LassoRegressionCustom, "params": {"alpha": 0.01}, "type": "regr"},
    #{"model": RidgeRegressionCustom, "params": {"alpha": 0.01}, "type": "regr"},
    #{"model": Lasso, "params": {"alpha": 0.01}, "type": "regr"},
    #{"model": Ridge, "params": {"alpha": 0.01}, "type": "regr"},
    {"model":SVC, "params": {"kernel":"linear","random_state":42}, "type": "class"},
    {"model":SupportVectorMachineCustom, "params": {"learning_rate":0.005,"lambda_param":0.01,"n_iters":2000}, "type": "class"},
    {"model": DecisionTreeClassifier, "params": {}, "type": "class"}

>>>>>>> b695b5fa35bbe2a7a3e07fe7a236b9257c420d9b
]

scalers = [Normalizer(), MinMaxScaler(), StandardScaler(), QuantileTransformer()]

class Runner:
    def __init__(self):
        self.model_trainer = ModelTrainer()

    def run(self):
        x_train, y_train, x_test, y_true = self.model_trainer.process_data()
        results = []

        best_models = {}  # Dictionnaire pour stocker le meilleur modèle de chaque type

        for model_info in model_list:
            model_class = model_info["model"]
            param_dist = model_info["params"]
            type_model = model_info["type"]

<<<<<<< HEAD
            for scaler in scalers:
                x_train_scaled = scaler.fit_transform(X=x_train)
                x_test_scaled = scaler.transform(X=x_test)  # Utiliser transform ici

                for param_values in self._get_param_combinations(param_dist):
                    # Instantiate the model with the current parameter values
                    model_instance = model_class(**param_values)

                    model_instance.fit(x_train_scaled, y_train)  # Fit the model
                    y_pred = model_instance.predict(x_test_scaled)  # Utiliser les données de test

                    # Calculate metrics
                    if type_model == "class":
                        acc = accuracy_score(y_true, y_pred)
                        # Update the best model if current accuracy is better
                        if type_model not in best_models or acc > best_models[type_model]['Accuracy']:
                            best_models[type_model] = {
                                "Model": model_info['model'].__name__,
                                "Scaler": scaler.__class__.__name__,
                                "Params": param_values,
                                "Accuracy": acc
                            }
=======
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
                f1 = f1_score(y_true, y_pred, average='weighted')
                conf_mat = confusion_matrix(y_true, y_pred)
                results.append({
                    "Model": nameModel,
                    "accuracy": acc,
                    "f1_score": f1,
                    "time": elapsed
                })
                # Plot the confusion matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'Confusion Matrix - {nameModel}')
                plt.savefig(f'figs/confusion_matrix_{nameModel}.png')
                print(f"Confusion matrix for {nameModel} exported to 'confusion_matrix_{nameModel}.png'.")
>>>>>>> b695b5fa35bbe2a7a3e07fe7a236b9257c420d9b

        # Export the best model information to a CSV file
        best_models_df = pd.DataFrame(best_models.values())
        best_models_df.to_csv("best_model_results.csv", index=False)
        print("Best models results exported to 'best_model_results.csv'.")

<<<<<<< HEAD
    def _get_param_combinations(self, param_dist):
        """Génère toutes les combinaisons possibles de paramètres."""
        from itertools import product
        keys = param_dist.keys()
        values = [param_dist[key] for key in keys]
        return [dict(zip(keys, v)) for v in product(*values)]
=======
                # Plotting the correlation matrix using seaborn
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
                plt.title('Correlation Matrix of Training Features')
                plt.savefig(f"figs/corrMat.png")
                print("Correlation matrix exported to 'corrMat.png'.")

        # Create a DataFrame from the results and export to CSV
        results_df = pd.DataFrame(results)
        if os.path.isfile("model_comparison_results.csv"):
            os.remove("model_comparison_results.csv")
        results_df.to_csv("model_comparison_results.csv", index=False)
        print("Model comparison results exported to 'model_comparison_results.csv'.")
>>>>>>> b695b5fa35bbe2a7a3e07fe7a236b9257c420d9b


if __name__ == "__main__":
    runner = Runner()
    runner.run()
