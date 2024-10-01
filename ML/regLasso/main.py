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
        """
        Entraîne le modèle LASSO sur les données.

        :param X: Matrice des caractéristiques
        :param y: Vecteur des cibles
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Normalisation des caractéristiques
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        for _ in range(self.iterations):
            # Calcul des prédictions
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Erreurs
            error = y_predicted - y
            
            # Mise à jour des poids et biais
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Mise à jour des poids avec régularisation L1
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Appliquer la régularisation L1 (LASSO)
            for j in range(n_features):
                self.weights[j] = self._soft_threshold(self.weights[j], self.alpha * self.learning_rate)

    def predict(self, X):
        """
        Prédit les valeurs pour un jeu de données donné.

        :param X: Matrice des caractéristiques
        :return: Prédictions
        """
        # Normalisation des caractéristiques avant la prédiction
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        return np.dot(X, self.weights) + self.bias


# if __name__ == "__main__":
#     X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
#     y = np.array([5, 7, 9, 11])

#     lasso = LassoRegression(alpha=0.1, learning_rate=0.01, iterations=1000)
#     lasso.fit(X, y)

#     predictions = lasso.predict(X)
#     print("Prédictions:", predictions)
