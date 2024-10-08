import itertools  # Importer itertools pour les combinaisons de paramètres
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import (MinMaxScaler, Normalizer,
                                   QuantileTransformer, StandardScaler)
from sklearn.tree import DecisionTreeClassifier

# Define project paths
PROJECT_PATH = Path(__file__).parents[1]
sys.path.append(str(PROJECT_PATH))
SRC_PATH = Path(__file__).resolve().parents[0]
sys.path.append(str(SRC_PATH))

# Imports custom functions
from ML.arbres.main import DecisionTree
from Pretreatment.ModelTrainer import ModelTrainer

model_list = [
    {"model": DecisionTreeClassifier, "params": {"max_depth": [1, 2, 3, 4, 5], "criterion": ["gini", "entropy"]}, "type": "class"},
    {"model": DecisionTree, "params": {"max_depth": [1, 2, 3, 4, 5]}, "type": "class"}
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
