import numpy as np
from sklearn import tree

class Foret() :

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, min_weight_fraction_leaf=0, max_features="auto",
                max_leaf_nodes=None, min_impurity_decrease=0, random_state=None, 
                class_weight=None, max_sample=None, max_columns=4):
        
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._min_weight_fraction_leaf = min_weight_fraction_leaf
        self._max_features = max_features
        self._max_leaf_nodes = max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        self._random_state = random_state
        self._class_weight = class_weight
        self._max_sample = max_sample
        self._max_columns = max_columns
        self._model = dict()
        self._features = dict()

    def fit(self, x, y):
 
        if not(self._max_sample) :
            self._max_sample = x.shape[0]
        
        for est in range(self._n_estimators) :
            rand_index = np.random.randint(low=0, high=x.shape[0], size=self._max_sample)
        
            rand_column = np.random.randint(low=0, high=x.shape[1], size=self._max_columns)
        
            x_samp = x[rand_index, :]
            x_samp = x_samp[:, rand_column]
        
            y_samp = y[rand_index]
        
            decision_tree_model = tree.DecisionTreeClassifier(max_depth=self._max_depth, 
                                                        min_samples_split=self._min_samples_split,
                                                        min_samples_leaf=self._min_samples_leaf,
                                                        min_weight_fraction_leaf=self._min_weight_fraction_leaf,
                                                        max_features=self._max_features,
                                                        max_leaf_nodes=self._max_leaf_nodes,
                                                        min_impurity_decrease=self._min_impurity_decrease,
                                                        random_state=self._random_state,
                                                        class_weight=self._class_weight)
            self._model[est] = decision_tree_model.fit(x_samp, y_samp)
            self._features[est] = rand_column


    def predict(self, x):
 
        pred = np.zeros(x.shape[0])

        for i in range(self._n_estimators):
            pred += self._model[i].predict(x[:, self._features[i]])
        
        pred = pred / self._n_estimators
        print(pred)
        pred = np.where(pred &gt;= 0.5, 1, 0)
        
        return pred

