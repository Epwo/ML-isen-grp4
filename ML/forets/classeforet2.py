from arbre import KDThree
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


class RandomForest(KDThree):
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
            tree = KDThree()
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
    

df = pd.read_csv('data\Carseats.csv')

X = df.drop(['Unnamed: 0','High'],axis= 1)
X = pd.get_dummies(X)
y = df['High']

le = LabelEncoder()
y_encoded = pd.Series(le.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

rf = RandomForest(n_estimators=100)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy fait maison: ", accuracy)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy sklearn: ", accuracy)