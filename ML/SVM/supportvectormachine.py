import matplotlib.pyplot as plt
import numpy as np


class SupportVectorMachineCustom:
    def __init__(self, learning_rate=0.005, lambda_param=0.01, n_iters=30000):
        # initialisation des paramètres d'apprentissage
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y, batch_size=32, decay=0.001):
        n_samples, n_features = X.shape
        # Initialisation des poids (matrice) et du biais
        self.w = np.zeros(n_features)
        self.b = 0
        y_ = np.where(y <= 0, -1, 1)

        # Nombre total de mini-batchs
        n_batches = n_samples // batch_size
        # Boucle principale sur les itérations
        for g in range(self.n_iters):
            # Réduction progressive du learning rate
            lr = self.learning_rate / (1 + decay * g)

            for i in range(n_batches):
                # Sélection d'un mini-batch aléatoire
                idx = np.random.randint(0, n_samples, batch_size)
                X_batch, y_batch = X[idx], y_[idx]

                # Calcul du produit matriciel pour le mini-batch
                margin = y_batch * (np.dot(X_batch, self.w) - self.b)

                # Sélection des exemples mal classés dans le mini-batch
                misclassified = np.where(margin < 1)[0]
                if misclassified.size > 0:
                    X_misclassified = X_batch[misclassified]
                    y_misclassified = y_batch[misclassified]

                    # Mise à jour des poids et du biais pour les exemples mal classés
                    self.w -= lr * (2 * self.lambda_param * self.w - np.dot(X_misclassified.T, y_misclassified))
                    self.b -= lr * np.sum(y_misclassified)

                # Réduction des poids pour les exemples bien classés (régularisation L2)
                else:
                    self.w -= lr * 2 * self.lambda_param * self.w


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
