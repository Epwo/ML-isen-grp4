import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth  # Profondeur maximale de l'arbre
        self.tree = None  # L'arbre sera construit lors de l'entraînement

    def get_params(self, deep=False):
        """Retourne les paramètres de l'arbre de décision."""
        return {
            'max_depth': self.max_depth
        }
    def set_params(self, **params):
        """Met à jour les paramètres de l'arbre de décision."""
        for key, value in params.items():
            if key == 'max_depth':
                self.max_depth = value
            else:
                raise ValueError(f"Le paramètre '{key}' n'est pas reconnu.")
    
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values  # Convertir en np.array si nécessaire
        if isinstance(y, pd.Series):
            y = y.values  # Convertir en np.array si nécessaire
            
        if len(X) != len(y):
            raise ValueError("Les dimensions de X et y doivent correspondre.")

        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Les données ne doivent pas contenir de valeurs manquantes.")

        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        # Si la profondeur maximale est atteinte ou si toutes les étiquettes sont identiques
        if depth >= self.max_depth or len(set(y)) == 1:
            return {'label': self._majority_vote(y)}  # Retourne la classe majoritaire

        n_samples, n_features = X.shape
        best_gain = 0
        best_split = None
        best_left_indices = None
        best_right_indices = None

        # Trouver la meilleure division
        for feature_index in range(n_features):
            thresholds, classes = zip(*sorted(zip(X[:, feature_index], y)))

            for i in range(1, n_samples):  # Éviter de diviser en deux parties vides
                if thresholds[i] == thresholds[i - 1]:
                    continue

                left_indices = np.where(X[:, feature_index] < thresholds[i])[0]
                right_indices = np.where(X[:, feature_index] >= thresholds[i])[0]

                gain = self._information_gain(y, left_indices, right_indices)

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, thresholds[i])
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        if best_gain == 0:  # Aucun gain d'information
            return {'label': self._majority_vote(y)}  # Retourne la classe majoritaire

        # Créer des sous-arbres
        left_subtree = self._build_tree(X[best_left_indices], y[best_left_indices], depth + 1)
        right_subtree = self._build_tree(X[best_right_indices], y[best_right_indices], depth + 1)

        return {
            'feature_index': best_split[0],
            'threshold': best_split[1],
            'left': left_subtree,
            'right': right_subtree
        }

    def _information_gain(self, y, left_indices, right_indices):
        p = len(left_indices) / (len(left_indices) + len(right_indices))
        return self._entropy(y) - p * self._entropy(y[left_indices]) - (1 - p) * self._entropy(y[right_indices])

    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log(probabilities + 1e-9))  # Ajout d'un petit nombre pour éviter log(0)

    def _majority_vote(self, y):
        """Retourne la classe majoritaire dans y."""
        return np.bincount(y).argmax()  # Retourne l'étiquette la plus fréquente

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return np.array([self._predict_one(self.tree, point) for point in X])

    def _predict_one(self, node, point):
        """Prédit la classe pour un seul point."""
        if 'label' in node:
            return node['label']

        feature_index = node['feature_index']
        threshold = node['threshold']

        if point[feature_index] < threshold:
            return self._predict_one(node['left'], point)
        else:
            return self._predict_one(node['right'], point)

# Tests
if __name__ == "__main__":
    data = {
        'Température': ['Chaud', 'Chaud', 'Froid', 'Froid', 'Chaud', 'Froid', 'Chaud'],
        'Humidité': ['Haute', 'Haute', 'Normale', 'Normale', 'Normale', 'Haute', 'Normale'],
        'Vent': ['Non', 'Oui', 'Non', 'Oui', 'Oui', 'Oui', 'Non'],
        'Jouer': ['Non', 'Non', 'Oui', 'Oui', 'Oui', 'Non', 'Oui']
    }
    df = pd.DataFrame(data)

    X = df.drop(columns='Jouer')
    y = df['Jouer'].map({'Non': 0, 'Oui': 1})  # Convertir les étiquettes en entiers

    X_encoded = pd.get_dummies(X, drop_first=True)

    # Initialiser et entraîner l'arbre de décision avec une profondeur maximale de 2
    dt = DecisionTree(max_depth=2)
    dt.fit(X_encoded, y)

    new_data = pd.DataFrame({
        'Température': ['Chaud', 'Froid'],
        'Humidité': ['Normale', 'Haute'],
        'Vent': ['Oui', 'Non']
    })

    new_data_encoded = pd.get_dummies(new_data, drop_first=True)

    predictions = dt.predict(new_data_encoded)
    print(predictions)