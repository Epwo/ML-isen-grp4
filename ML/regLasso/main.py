import numpy as np


class LassoRegression:
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
        Fonction de seuillage douce pour imposer la pénalité l1.
        
        :param rho: Valeur à ajuster
        :param alpha: Paramètre de régularisation
        :return: Valeur ajustée
        """
        if rho < -alpha:
            return rho + alpha
        elif rho > alpha:
            return rho - alpha
        else:
            return 0

    def fit(self, X, y):
        """
        Entraîne le modèle sur les données d'entraînement.
        
        :param X: Matrice des caractéristiques
        :param y: Vecteur des cibles
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            dw = np.dot(X.T, (y_predicted - y)) / n_samples
            db = np.sum(y_predicted - y) / n_samples
            
            for j in range(n_features):
                z = self.weights[j] - self.learning_rate * dw[j]
                self.weights[j] = self._soft_threshold(z, self.alpha * self.learning_rate)
            
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Prédit les valeurs cibles pour un jeu de données donné.
        
        :param X: Matrice des caractéristiques
        :return: Prédictions
        """
        return np.dot(X, self.weights) + self.bias

# if __name__ == "__main__":
#     X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
#     y = np.array([5, 7, 9, 11])

#     lasso = LassoRegression(alpha=0.1, learning_rate=0.01, iterations=1000)
#     lasso.fit(X, y)

#     predictions = lasso.predict(X)
#     print("Prédictions:", predictions)
