from arbre import KDTree
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class RandomForest(KDTree):
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__()
        # nombre d'arbres
        self.n_estimators = n_estimators
        # profondeur maximale pour les arbres
        self.max_depth = max_depth
        # liste contenant les arbres de la forêt
        self.trees = []

    def fit(self, X, y):
        # pour chaque arbre
        for _ in range(self.n_estimators):
            # formation d'échantillons par sélection aléatoire (avec remplacement) des données d'entraînement
            # index de sélection des données
            idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            # caractéristiques de prédiction
            X_boot = X.iloc[idx]
            # caractéristique à prédire
            y_boot = y.iloc[idx]

            # sélection aléatoire d'un sous ensemble de caractéristiques utilisées pour l'entraînement de l'arbre
            # remarque : majoration de la taille de ce sous ensemble pour réduire les temps de calcul et diversifier les arbres
            feature_subset = np.random.choice(X.columns, int(np.sqrt(X.shape[1])), replace=False)

            # entraînement de l'arbre
            # tree = KDTree()
            # tree.fit(X_boot[feature_subset], y_boot)

            tree = DecisionTreeClassifier()
            tree.fit(X_boot[feature_subset], y_boot)

            # ajout de l'abre à la forêt
            self.trees.append(tree)
    
    def predict(self, X):

        # la prédiction de la forêt est définie comme la valeur apparaissant le plus de fois dans les prédictions de ses arbres
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        # transposée permet d'obtenir le bon format (arbres en colonne et échantillons en ligne)
        predictions = np.array(predictions).T

        return pd.Series([pd.Series(p).mode().iloc[0] if not pd.Series(p).isnull().all() else np.nan for p in predictions])
    

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

    # Initialiser et entraîner l'arbre KD avec une profondeur maximale de 2
    dt = KDTree(max_depth=2, k=3)  # Utiliser les 3 voisins les plus proches
    dt.fit(X_encoded, y)

    rf = RandomForest(n_estimators=100)
    rf.fit(X_encoded, y)

    new_data = pd.DataFrame({
        'Température': ['Chaud', 'Froid'],
        'Humidité': ['Normale', 'Haute'],
        'Vent': ['Oui', 'Non']
    })

    new_data_encoded = pd.get_dummies(new_data, drop_first=True)

    predictions = dt.predict(new_data_encoded)
    print(predictions)

    predictions = rf.predict(new_data_encoded)
    print(predictions)


# df = pd.read_csv('data\Carseats.csv')

# X = df.drop(['Unnamed: 0','High'],axis= 1)
# X = pd.get_dummies(X)

# y = df['High'].apply(lambda x : 1 if x == 'Yes' else 0)
# X[X.columns] = X[X.columns].astype(int)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# rf = RandomForest(n_estimators=100)
# rf.fit(X_train, y_train)
# predictions = rf.predict(X_test)

# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy fait maison: ", accuracy)

# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy sklearn: ", accuracy)