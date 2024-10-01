from main import RidgeRegression
import numpy as np
# Création d'un dataset factice

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(100)

# Initialisation et entraînement du modèle de régression Ridge
ridge_reg = RidgeRegression(alpha=1.0)
ridge_reg.fit(X, y)

# Prédictions
y_pred = ridge_reg.predict(X)

# Affichage des résultats
print("Coefficients:", ridge_reg.weights)
print("Prédictions pour les 5 premiers échantillons:", y_pred[:5])
