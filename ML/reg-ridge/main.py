import numpy as np


class RidgeRegression:
    # pour tout les codes suivants, nous allons utiliser l'opétateur @, qui est l'opérateur de multiplication de matrice
    def __init__(self, alpha=1.0):
        """
        Initialise un modèle de ridge regression.
        avec alpha le coeff de régularisation
        """
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        train le modele de régression Ridge sur les données.
        X : matrice de caractéristiques (n_samples, n_features)
        y : vecteur cible (n_samples,)
        """
        n_samples, n_features = X.shape

        # Ajouter une colonne de biais
        X_b = np.hstack([np.ones((n_samples, 1)), X])

        # Calcul des coefficients avec régularisation L2
        I = np.eye(n_features + 1)
        I[0, 0] = 0  # Ne pas régulariser le biais
        self.weights = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y

    def predict(self, X):
        """
        avec le modele entrainé dans self, il va prédir les valeurs de X
        avec X matrice de caractéristiques (n_samples, n_features)
        """
        n_samples = X.shape[0]

        # Ajouter une colonne de biais
        bias_column = np.ones((n_samples, 1))  # Créer une colonne de 1 pour le biais
        X_b = np.hstack([bias_column, X])  # Ajouter cette colonne à gauche de X

        return X_b @ self.weights
