import numpy as np


class LassoRegressionCustom:
    def __init__(self, alpha=0.01, learning_rate=0.01, iterations=1000):
        """
        Initialisation du modèle LASSO

        :param alpha: Coefficient de régularisation (lambda)
        :param learning_rate: Taux d'apprentissage pour la descente de gradient
        :param iterations: Nombre d'itérations pour la descente de gradient
        """
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def _soft_threshold(self, rho, alpha):
        """
        Fonction de seuillage doux pour la régularisation L1 (LASSO).

        :param rho: Valeur à ajuster
        :param alpha: Paramètre de régularisation
        :return: Valeur ajustée
        """
        if rho > alpha:
            return rho - alpha
        elif rho < -alpha:
            return rho + alpha
        else:
            return 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Normalisation des caractéristiques (moyenne et écart type)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_normalized = (X - self.mean) / self.std
        
        for _ in range(self.iterations):
            y_predicted = np.dot(X_normalized, self.weights) + self.bias
            error = y_predicted - y
            
            dw = (1 / n_samples) * np.dot(X_normalized.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Mise à jour des poids
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Régularisation L1
            for j in range(n_features):
                self.weights[j] = self._soft_threshold(self.weights[j], self.alpha * self.learning_rate)

    def predict(self, X):
        X_normalized = (X - self.mean) / self.std  # Utiliser la moyenne et l'écart type calculés lors de l'entraînement
        return np.dot(X_normalized, self.weights) + self.bias