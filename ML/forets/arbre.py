import numpy as np
import pandas as pd
import math
# from classeforet2 import RandomForest
from sklearn.tree import DecisionTreeClassifier

class KDTree:
    def __init__(self, max_depth=None, k=2):
        self.tree = None
        self.max_depth = max_depth  # Profondeur maximale de l'arbre
        self.k = k  # Nombre de voisins à considérer pour la prédiction

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
        # Vérifier si la profondeur maximale est atteinte
        if self.max_depth is not None and depth >= self.max_depth:
            return {'label': self._majority_vote(y)}  # Retourner la classe majoritaire

        if len(set(y)) == 1:
            return {'label': y[0]}  # Retourner un dictionnaire avec la seule étiquette

        if len(X) == 0:
            return None  # Si aucun point, retourner None

        axis = depth % X.shape[1]  # Choisir l'axe basé sur la profondeur

        sorted_indices = np.argsort(X[:, axis])
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]

        median_index = len(X_sorted) // 2

        node = {
            'point': X_sorted[median_index],
            'label': y_sorted[median_index],
            'left': self._build_tree(X_sorted[:median_index], y_sorted[:median_index], depth + 1),
            'right': self._build_tree(X_sorted[median_index + 1:], y_sorted[median_index + 1:], depth + 1),
            'axis': axis
        }

        return node

    def _majority_vote(self, y):
        """Retourne la classe majoritaire dans y."""
        return np.bincount(y).argmax()  # Retourne l'étiquette la plus fréquente

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return np.array([self._predict_one(self.tree, point) for point in X])

    def _predict_one(self, node, point):
        """Prédit la classe pour un seul point."""
        if node is None:
            return None
        
        # Initialiser la liste des voisins trouvés
        neighbors = []
        self._search_nearest_neighbors(node, point, neighbors)

        # Récupérer les étiquettes des voisins trouvés
        labels = [neighbor['label'] for neighbor in neighbors]
        return self._majority_vote(labels)  # Retourner la classe majoritaire des voisins

    def _search_nearest_neighbors(self, node, point, neighbors, depth=0):
        if node is None:
            return

        # Vérifiez si le nœud est une feuille ou un nœud interne
        if 'point' in node:
            # Calculer la distance à ce nœud
            # print(point)
            # print('-----------------')
            # print(node['point'])
            # print('-----------------')
            distance = np.linalg.norm(point ^ node['point'])
            
            # Ajouter le nœud courant aux voisins
            neighbors.append({'point': node['point'], 'label': node['label'], 'distance': distance})

            # Si nous avons déjà k voisins, trions par distance
            if len(neighbors) > self.k:
                neighbors.sort(key=lambda x: x['distance'])
                neighbors.pop()  # Retirer le voisin le plus éloigné

            # Vérifier dans quel sous-arbre nous devrions aller
            axis = node['axis']
            if point[axis] < node['point'][axis]:
                self._search_nearest_neighbors(node['left'], point, neighbors, depth + 1)
                if len(neighbors) < self.k or abs(point[axis] ^ node['point'][axis]) < neighbors[-1]['distance']:
                    self._search_nearest_neighbors(node['right'], point, neighbors, depth + 1)
            else:
                self._search_nearest_neighbors(node['right'], point, neighbors, depth + 1)
                if len(neighbors) < self.k or abs(point[axis] ^ node['point'][axis]) < neighbors[-1]['distance']:
                    self._search_nearest_neighbors(node['left'], point, neighbors, depth + 1)
        else:
            # Si c'est un nœud feuille, nous ajoutons seulement l'étiquette
            neighbors.append({'label': node['label'], 'distance': float('inf')})  # Utiliser une distance infinie pour le nœud feuille

class RandomForest(KDTree):
    def __init__(self, n_estimators=100, max_depth=None, max_features = 'sqrt',bootstrap = True):
        super().__init__()
        # nombre d'arbres
        self.n_estimators = n_estimators
        # profondeur maximale pour les arbres
        self.max_depth = max_depth
        # nombre maximal de caractéristiques à considérer pour chaque arbre
        self.max_features = max_features
        # utilisation d'un bootstrap ou non (si True alors entraînement des arbres sur échantillons, False --> entraînement sur tout le dataset)
        self.bootstrap = bootstrap
        # liste contenant les arbres de la forêt
        self.trees = []

    def fit(self, X, y):
        # pour chaque arbre
        for _ in range(self.n_estimators):

            if self.bootstrap == True : 
                # formation d'échantillons par sélection aléatoire (avec remplacement) des données d'entraînement
                # index de sélection des données
                idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
                # caractéristiques de prédiction
                X = X.iloc[idx]
                # caractéristique à prédire
                y = y.iloc[idx]

            # sélection aléatoire d'un sous ensemble de caractéristiques utilisées pour l'entraînement de l'arbre
            # remarque : majoration de la taille de ce sous ensemble pour réduire les temps de calcul et diversifier les arbres
            if self.max_features == 'sqrt' : 
                feature_subset = np.random.choice(X.columns, int(np.sqrt(X.shape[1])), replace=False)
            elif self.max_features == 'log2' : 
                feature_subset = np.random.choice(X.columns, int(math.log2(X.shape[1])), replace=False)
            elif self.max_features == None : 
                feature_subset = np.random.choice(X.columns, int(X.shape[1]), replace=False)

            # entraînement de l'arbre
            tree = KDTree(max_depth = self.max_depth)
            tree.fit(X[feature_subset], y)

            # tree = DecisionTreeClassifier()
            # tree.fit(X_boot[feature_subset], y_boot)

            # ajout de l'abre à la forêt
            self.trees.append(tree)
    
    def predict(self, X):

        # la prédiction de la forêt est définie comme la valeur apparaissant le plus de fois dans les prédictions de ses arbres
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        # transposée permet d'obtenir le bon format (arbres en colonne et échantillons en ligne)
        predictions = np.array(predictions).T
        # print(len(predictions[0]),len(predictions[1]),predictions)

        # print(pd.Series([pd.Series(p).mode().iloc[0] if not pd.Series(p).isnull().all() else np.nan for p in predictions]))

        return pd.Series([pd.Series(p).mode().iloc[0] if not pd.Series(p).isnull().all() else np.nan for p in predictions])

    # Tests
if __name__ == "__main__":

    data = {'Température' : [], 'Humidité' : [], 'Vent' : [], 'Jouer' : []}
    for i in range(10000) : 
        data['Température'].append(np.random.choice(['Froid','Chaud','Normale']))
        data['Humidité'].append(np.random.choice(['Haute','Normale']))
        data['Vent'].append(np.random.choice(['Oui','Non']))
        data['Jouer'].append(np.random.choice(['Oui','Non']))

    df = pd.DataFrame(data)

    X = df.drop(columns='Jouer')
    y = df['Jouer'].map({'Non': 0, 'Oui': 1})  # Convertir les étiquettes en entiers

    X_encoded = pd.get_dummies(X, drop_first=True)

    # Initialiser et entraîner l'arbre KD avec une profondeur maximale de 2
    dt = KDTree(k=3)  # Utiliser les 3 voisins les plus proches
    dt.fit(X_encoded, y)

    rf = RandomForest(n_estimators=100)
    rf.fit(X_encoded, y)

    new_data = pd.DataFrame({
        'Température': ['Chaud', 'Froid'],
        'Humidité': ['Normale', 'Haute'],
        'Vent': ['Oui', 'Non']
    })

    new_data_encoded = pd.get_dummies(new_data, drop_first=True)

    # predictions = dt.predict(new_data_encoded)
    # print(predictions)

    predictions = rf.predict(new_data_encoded)
    # print(predictions)
    # for pred in predictions :
    #     print(pred)

