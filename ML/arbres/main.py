import numpy as np
import pandas as pd

class KDThree:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        if len(set(y)) == 1:
            return y.iloc[0]

        if X.empty:
            return y.mode()[0]

        gains = []
        for col in X.columns:
            gain = self._information_gain(y, X[col])
            gains.append(gain)

        best_feature = X.columns[np.argmax(gains)]
        tree = {best_feature: {}}

        for value in X[best_feature].unique():
            subset_X = X[X[best_feature] == value].drop(columns=best_feature)
            subset_y = y[X[best_feature] == value]
            tree[best_feature][value] = self._build_tree(subset_X, subset_y)

        return tree

    def _entropy(self, y):
        prob = y.value_counts(normalize=True)
        return -sum(prob * np.log2(prob + 1e-9))

    def _information_gain(self, y, X_col):
        total_entropy = self._entropy(y)
        values = X_col.unique()
        weighted_entropy = 0

        for value in values:
            subset_y = y[X_col == value]
            weighted_entropy += (len(subset_y) / len(y)) * self._entropy(subset_y)

        return total_entropy - weighted_entropy

    def predict(self, X):
        return X.apply(self._predict_one, axis=1)

    def _predict_one(self, row):
        tree = self.tree
        while isinstance(tree, dict):
            feature = next(iter(tree))
            value = row[feature]
            tree = tree[feature].get(value, None)
        return tree

#Tests
if __name__ == "__main__":
    data = {
        'Température': ['Chaud', 'Chaud', 'Froid', 'Froid', 'Chaud', 'Froid', 'Chaud'],
        'Humidité': ['Haute', 'Haute', 'Normale', 'Normale', 'Normale', 'Haute', 'Normale'],
        'Vent': ['Non', 'Oui', 'Non', 'Oui', 'Oui', 'Oui', 'Non'],
        'Jouer': ['Non', 'Non', 'Oui', 'Oui', 'Oui', 'Non', 'Oui']
    }
    df = pd.DataFrame(data)

    X = df.drop(columns='Jouer')
    y = df['Jouer']

    dt = KDThree()
    dt.fit(X, y)

    new_data = pd.DataFrame({
        'Température': ['Chaud', 'Froid', 'Chaud'],
        'Humidité': ['Normale', 'Haute', 'Normale'],
        'Vent': ['Oui', 'Non', 'Oui']
    })
    predictions = dt.predict(new_data)
    print(predictions)
