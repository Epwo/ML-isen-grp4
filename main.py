import itertools  # Importer itertools pour les combinaisons de paramètres
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import (MinMaxScaler, Normalizer,
                                   QuantileTransformer, StandardScaler)

# Define project paths
PROJECT_PATH = Path(__file__).parents[1]
sys.path.append(str(PROJECT_PATH))
SRC_PATH = Path(__file__).resolve().parents[0]
sys.path.append(str(SRC_PATH))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Imports custom functions
from ML.arbres.main import DecisionTree
# imports customs functions
from ML.forets.arbre import DecisionTree
from ML.forets.foret import RandomForest as RandomForestCustom
from ML.regLasso.main import LassoRegressionCustom
from ML.regRidge.main import RidgeRegressionCustom
from ML.SVM.supportvectormachine import SupportVectorMachineCustom
from Pretreatment.ModelTrainer import ModelTrainer

# model_list = [
#     {"model": LassoRegressionCustom, 
#      "params": {"alpha": [0.01, 0.1, 1, 10, 100]}, 
#      "type": "regr"},
    
#     {"model": RidgeRegressionCustom, 
#      "params": {"alpha": [0.01, 0.1, 1, 10, 100]}, 
#      "type": "regr"},
    
#     {"model": Lasso, 
#      "params": {"alpha": [0.01, 0.1, 1, 10, 100]}, 
#      "type": "regr"},
    
#     {"model": Ridge, 
#      "params": {"alpha": [0.01, 0.1, 1, 10, 100]}, 
#      "type": "regr"}
# ]

model_list = [
    {"model": DecisionTreeClassifier, 
     "params": {"max_depth": [1, 2, 3, 4, 5, 10], "criterion": ["gini", "entropy"]}, 
     "type": "class"},
    
    {"model": DecisionTree, 
     "params": {"max_depth": [1, 2, 3, 4, 5, 10]}, 
     "type": "class"},
    
    {"model": SVC, 
     "params": {"kernel": ["linear", "rbf", "poly"], "random_state": [42, 123, 2024]}, 
     "type": "class"},
    
    {"model": SupportVectorMachineCustom, 
     "params": {"learning_rate": [0.005], "lambda_param": [0.01], "n_iters": [2000]}, 
     "type": "class"},
    
    {"model": RandomForestClassifier, "params": {"n_estimators": [100], "max_depth": [2]}, "type": "class"},

    {"model": RandomForestCustom, "params": {"n_estimators": [100], "max_depth": [2]}, "type": "class"}
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
            print(f"Train : {model_info['model'].__name__}")
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
                        f1 = f1_score(y_true, y_pred, average='weighted')
                        # Update the best model if current accuracy is better
                        if type_model not in best_models or acc > best_models[model_info['model'].__name__]['Accuracy']:
                            best_models[model_info['model'].__name__] = {
                                "Model": model_info['model'].__name__,
                                "Scaler": scaler.__class__.__name__,
                                "Params": param_values,
                                "Accuracy": acc,
                                "accuracy": acc,
                                "f1_score": f1
                            }

                    if model_info["type"] == "regr":
                        # Store results
                        mae = mean_absolute_error(y_true, y_pred)
                        mse = mean_squared_error(y_true, y_pred)
                        r2 = r2_score(y_true, y_pred)
                        if type_model not in best_models or r2 > best_models[model_info['model'].__name__]['R2_Score']:
                            best_models[model_info['model'].__name__] = {
                                "Model": model_info['model'].__name__,
                                "MAE": mae,
                                "MSE": mse,
                                "R2_Score": r2
                            }

        # Export the best model information to a CSV file
        best_models_df = pd.DataFrame(best_models.values())
        best_models_df.to_csv("best_model_results.csv", index=False)
        print("Best models results exported to 'best_model_results.csv'.")

    def _get_param_combinations(self, param_dist):
        """Génère toutes les combinaisons possibles de paramètres."""
        from itertools import product
        keys = param_dist.keys()
        values = [param_dist[key] for key in keys]
        return [dict(zip(keys, v)) for v in product(*values)]
if __name__ == "__main__":
    runner = Runner()
    runner.run()