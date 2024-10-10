
import numpy as np
import pandas as pd


# implémenter le warm_start et random state
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
        # print(X) 
        # print(y)
        # if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        #     raise ValueError("Les données ne doivent pas contenir de valeurs manquantes.")

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
            

class RandomForest(DecisionTree):
    def __init__(self, n_estimators=100, max_depth=2, max_features='sqrt', bootstrap=True, warm_start=False, random_state=None):
        super().__init__()
        # nombre d'arbres
        self.n_estimators = n_estimators
        # profondeur maximale pour les arbres
        self.max_depth = max_depth
        # nombre maximal de caractéristiques à considérer pour chaque arbre
        self.max_features = max_features
        # utilisation d'un bootstrap ou non (si True alors entraînement des arbres sur échantillons, False --> entraînement sur tout le dataset)
        self.bootstrap = bootstrap
        # warm_start  = False fit une nouvelle forêt, sinon ajoute de nouveaux arbres à la forêt créée lors du précédent appel
        self.warm_start = warm_start
        # warm_start  = False fit une nouvelle forêt, sinon ajoute de nouveaux arbres à la forêt créée lors du précédent appel
        self.random_state = np.random.RandomState(random_state) if random_state is not None else None
        # liste contenant les arbres de la forêt
        self.trees = []
        # dictionnaire contenant les numéros des arbres de la forêt et leurs caractéristiques d'expertise
        self.dic = {}

    def check_params(self):
        if type(self.n_estimators) != int or self.n_estimators < 1:
            raise ValueError("Le paramètre n_estimators doit être un entier supérieur à zéro")
        if type(self.max_depth) != int or self.max_depth < 1:
            raise ValueError("Le paramètre max_depth doit être un entier supérieur à zéro")
        if self.max_features not in ['sqrt', 'log2', None]:
            raise ValueError("Le paramètre max_features doit prendre une des valeurs comprise dans la liste ['sqrt', 'log2', None]")
        if type(self.bootstrap) != bool:
            raise ValueError("Le paramètre bootstrap doit être un booléen")
        if type(self.warm_start) != bool:
            raise ValueError("Le paramètre warm_start doit être un booléen")
        if self.random_state is not None and not isinstance(self.random_state, np.random.RandomState):
            raise ValueError("Le paramètre random_state doit être égal à None ou un objet RandomState")

    def fit(self, X, y):

        # Vérification des paramètres
        self.check_params()

        # Conversions en tableau numpy
        X = np.array(X)  
        y = np.array(y)

        # pour chaque arbre
        for i in range(self.n_estimators):

            if self.bootstrap:
                # formation d'échantillons par sélection aléatoire (avec remplacement) des données d'entraînement
                idx = self.random_state.choice(X.shape[0], X.shape[0], replace=True) if self.random_state else np.random.choice(X.shape[0], X.shape[0], replace=True)
                X_sample, y_sample = X[idx], y[idx]
            else:
                # échantillons = ensemble du dataset
                X_sample, y_sample = X, y

            # sélection aléatoire d'un sous ensemble (de taille paramétrable) de caractéristiques utilisées pour l'entraînement de l'arbre
            if self.max_features == 'sqrt':
                feature_subset = self.random_state.choice(X.shape[1], int(np.sqrt(X.shape[1])), replace=False) if self.random_state else np.random.choice(X.shape[1], int(np.sqrt(X.shape[1])), replace=False)
            elif self.max_features == 'log2':
                feature_subset = self.random_state.choice(X.shape[1], int(np.log2(X.shape[1])), replace=False) if self.random_state else np.random.choice(X.shape[1], int(np.log2(X.shape[1])), replace=False)
            else:
                feature_subset = np.arange(X.shape[1])

            # Entraînement de l'arbre
            tree = DecisionTree()
            tree.fit(X_sample[:, feature_subset], y_sample)

            # Ajout de l'arbre à la forêt
            self.trees.append(tree)

            # enregistrement des features utilisées pour entraîner l'arbre i
            self.dic[i] = feature_subset

    def predict(self, X):

        # Conversion en tableau numpy
        X = np.array(X) 

        # la prédiction de la forêt est définie comme la valeur apparaissant le plus de fois dans les prédictions de ses arbres 
        predictions = []
        for i, tree in enumerate(self.trees):
            X_tmp = X[:, self.dic[i]]
            predictions.append(tree.predict(X_tmp))
        # transposée permet d'obtenir le bon format (arbres en colonne et échantillons en ligne)
        predictions = np.array(predictions).T

        # Calcul de la prédiction finale
        prediction = [int(np.bincount(pred).argmax()) for pred in predictions]
        return prediction
    


    
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    df = pd.read_csv("data\\Carseats.csv")

    X = df.drop(columns=['High','Unnamed: 0'])
    y = df['High'].map({'No': 0, 'Yes': 1})
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    import time
    start = time.time()
    ensemble_rf = RandomForest(n_estimators=50, max_features='log2')
    ensemble_rf.fit(X_train, y_train)
    predictions = ensemble_rf.predict(X_test)
    print(accuracy_score(y_test, predictions))
    print(time.time() - start)



