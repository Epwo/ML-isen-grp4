import numpy as np
import matplotlib.pyplot as plt


class SupportVectorMachineCustom:
    def __init__(self, learning_rate=0.005, lambda_param=0.01, n_iters=30000):
        # initialisation des paramètres d'apprentissage
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialisation des poids (matrice) et du biais à 0 ainsi que les features à -1 et 1
        self.w = np.zeros(n_features)
        self.b = 0
        y_ = np.where(y <= 0, -1, 1)

        # nb de fois qu'on effectue les opérations sur la BDD
        for g in range(self.n_iters):
            # itération sur chaque ligne de la BDD
            for idx, x_i in enumerate(X):

                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                # vérifie si la distance à la marge est respecté, on met alors à jour les poids
                # les poids sont légèrement diminuer
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                # les poids sont corrigés ainsi que le biais pour déplacer l'hyperplan
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        # Calcul de la valeur d'approximation pour chaque échantillon (produit scalaire + biais)
        approx = np.dot(X, self.w) - self.b
        # Renvoie 1 ou 0 si l'approximation est bonne ou non
        return np.where(approx >= 0, 1, 0)


# Normalisation Z-score
def z_score_normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Séparation des données test / données train
def train_test_split_custom(X, y, test_size=0.2):
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_idx, train_idx = indices[:test_size], indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def affichage(y_test, y_pred):
    # Visualisation des prédictions
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Vrais labels', alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Prédictions', alpha=0.6)
    plt.title('Comparaison des labels réels et des prédictions')
    plt.xlabel('Index')
    plt.ylabel('Label (0: Non, 1: Oui)')
    plt.legend(loc='best')
    plt.show()
