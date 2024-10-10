from arbre import DecisionTree
import numpy as np
import pandas as pd
import math


# notes : 

# implémenter le warm_start et random state

class RandomForest(DecisionTree):
    def __init__(self, n_estimators=100, max_depth=2, max_features = 'sqrt',bootstrap = True, warm_start = False, random_state = None):
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
        # random_state : influe sur le bootstaping (s'il est activé) ainsi que sur les caractéristiques 
        self.random_state = np.random.RandomState(random_state) if random_state is not None else None
        # liste contenant les arbres de la forêt
        self.trees = []
        # dictionnaire contenant les numéros des arbres de la forêt et leurs caractéristiques d'expertise
        self.dic = {}

    def check_params(self) :

        if type(self.n_estimators) != int or self.n_estimators < 1 :
            raise ValueError("Le paramètre n_estimators doit être un entier supérieur à zéro")
        if type(self.max_depth) != int or self.max_depth < 1 : 
            raise ValueError("Le paramètre max_depth doit être un entier supérieur à zéro")
        if self.max_features not in ['sqrt','log2',None] :
            raise ValueError("Le paramètre max_features doit prendre une des valeurs comprise dans la liste ['sqrt','log2',None] ")
        if type(self.bootstrap) != bool :
            raise ValueError("Le paramètre bootstrap doit être un booléen")
        if type(self.warm_start) != bool :
            raise ValueError("Le paramètre warm_start doit être un booléen")
        if type(self.random_state) != int and self.random_state != None :
            print(self.random_state)
            raise ValueError("Le paramètre random_state doit être égal à None ou un entier")        
        

    def fit(self, X, y):

        self.check_params()

        # pour chaque arbre
        for i in range(self.n_estimators):

            if self.bootstrap == True and self.random_state == None : 

                # formation d'échantillons par sélection aléatoire (avec remplacement) des données d'entraînement
                # index de sélection des données
                idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
                # caractéristiques de prédiction
                X = X.iloc[idx]
                # caractéristique à prédire
                y = y.iloc[idx]
                
            # elif self.bootstrap == True and self.random_state != None :
 
            #     if self.max_features == 'sqrt':
            #         feature_subset = self._random_state.choice(X.columns, int(np.sqrt(X.shape[1])), replace=False)
            #     elif self.max_features == 'log2':
            #         feature_subset = self._random_state.choice(X.columns, int(math.log2(X.shape[1])), replace=False)
            #     elif self.max_features == None:
            #         feature_subset = self._random_state.choice(X.columns, int(X.shape[1]), replace=False)


            # sélection aléatoire d'un sous ensemble (de taille paramétrable) de caractéristiques utilisées pour l'entraînement de l'arbre
            if self.max_features == 'sqrt' : 
                feature_subset = np.random.choice(X.columns, int(np.sqrt(X.shape[1])), replace=False)
            elif self.max_features == 'log2' : 
                feature_subset = np.random.choice(X.columns, int(math.log2(X.shape[1])), replace=False)
            elif self.max_features == None : 
                feature_subset = np.random.choice(X.columns, int(X.shape[1]), replace=False)

            # entraînement de l'arbre
            tree = DecisionTree()
            tree.fit(X[feature_subset], y)

            # ajout de l'abre à la forêt
            self.trees.append(tree)

            # enregistrement des features utilisées pour entraîner l'arbre i
            self.dic[i] = list(feature_subset)
    
    def predict(self, X):

        # la prédiction de la forêt est définie comme la valeur apparaissant le plus de fois dans les prédictions de ses arbres
        predictions = []
        for i , tree in enumerate(self.trees) :
            X_tmp = X[self.dic[i]]
            predictions.append(tree.predict(X_tmp))
        # transposée permet d'obtenir le bon format (arbres en colonne et échantillons en ligne)
        predictions = np.array(predictions).T

        prediction = []
        for pred in predictions :
            prediction.append(int(np.bincount(pred).argmax()))

        return prediction 
    
    
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    df = pd.read_csv("data\\Carseats.csv")

    X = df.drop(columns=['High','Unnamed: 0'])
    y = df['High'].map({'No': 0, 'Yes': 1})
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

    rf = RandomForest(n_estimators= 5, max_features='log2')
    rf.fit(X_train,y_train)

    predictions = rf.predict(X_test)

    print(accuracy_score(y_test, predictions))