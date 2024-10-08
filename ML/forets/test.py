import pandas as pd
from arbre import KDTree
from classeforet2 import RandomForest
from sklearn.ensemble import RandomForestClassifier
import numpy as np


clf = RandomForestClassifier()
# Utilisation de np.random.choice()

print(np.random.choice([1, 2, 3], size=5))

print(np.random.choice([1, 2, 3], size=5))  # résultats différents


# Utilisation de numpy.random.RandomState.choice()

rng = np.random.RandomState(42)

print(rng.choice([1, 2, 3], size=5))

print(rng.choice([1, 2, 3], size=5))

print("-----------------------")

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


# from classeRdForest import Foret
# import numpy as np
# import pandas as pds
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier


# df = pds.read_csv("C:\\Users\\ninop\\Documents\\ML-isen-grp4\\data\\Carseats.csv")

# np.random.seed(42)
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X[:, 0] + np.random.randn(100)

# X = df.drop(['Unnamed: 0','High'],axis= 1)
# X = pds.get_dummies(X)
# y = df['High']

# # foret = Foret()
# # foret.fit(X,y)
# # y_pred = foret.predict(X)

# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf.fit(X, y)
# y_pred = clf.predict(X)

# print(accuracy_score(y, y_pred))






