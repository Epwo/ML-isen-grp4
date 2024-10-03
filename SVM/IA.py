import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger le fichier CSV
file_path = '../data/Carseats.csv'  # Assurez-vous que le chemin est correct
df = pd.read_csv(file_path)

# Préparation des données
# Transformer la colonne 'High' en valeurs 0 et 1
df['High'] = df['High'].apply(lambda x: 1 if x == 'Yes' else 0)

# Sélectionner deux colonnes numériques pour l'entraînement
X = df[['CompPrice', 'Income']].values  # Deux caractéristiques
y = df['High'].values


# Implémentation d'un SVM basique from scratch
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None  # Poids
        self.b = None  # Biais

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = [0] * n_features  # Initialiser les poids à zéro
        self.b = 0  # Initialiser le biais à zéro

        for _ in range(self.n_iters):
            for idx in range(n_samples):
                x_i = X[idx]  # Obtenir l'échantillon actuel
                y_i = y[idx]  # Obtenir la vraie étiquette
                # Calculer la condition pour la mise à jour
                condition = y_i * (self.dot(x_i, self.w) - self.b)

                if condition < 1:  # Si l'échantillon est mal classé
                    # Mettre à jour les poids
                    for j in range(len(self.w)):  # Pour chaque poids
                        self.w[j] -= self.learning_rate * (
                                2 * self.lambda_param * self.w[j] - x_i[j] * y_i
                        )
                    # Mettre à jour le biais
                    self.b -= self.learning_rate * y_i
                else:  # Si l'échantillon est bien classé
                    # Mettre à jour les poids avec la pénalité de régularisation
                    for j in range(len(self.w)):
                        self.w[j] -= self.learning_rate * (2 * self.lambda_param * self.w[j])

    def predict(self, X):
        # Prédire la classe
        linear_output = [self.dot(x, self.w) - self.b for x in X]
        return [1 if output >= 0 else 0 for output in linear_output]

    def dot(self, x, w):
        # Produit scalaire
        return sum(x[i] * w[i] for i in range(len(x)))


# Instanciation du modèle
model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)

# Entraînement du modèle
model.fit(X, y)

# Prédiction avec toutes les données
y_pred = model.predict(X)

# Affichage des résultats
print("Prédictions:", y_pred[:5])
print("Vraies valeurs:", y[:5])

# Calcul de l'accuracy
accuracy = np.mean(y_pred == y) * 100
print(f"Accuracy: {accuracy:.2f}%")


# Visualisation de l'hyperplan
def plot_decision_boundary(X, y, model, y_pred):
    # Tracer les points
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='b', label='High = 1', marker='o', alpha=0.6)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='r', label='High = 0', marker='x', alpha=0.6)

    # Créer une grille pour tracer l'hyperplan
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Prédire pour chaque point de la grille
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)

    # Tracer les contours
    plt.contourf(xx, yy, Z, alpha=0.3)

    # Tracer l'hyperplan de décision
    plt.axhline(0, color='grey', lw=2)
    plt.axvline(0, color='grey', lw=2)

    # Titre et étiquettes
    plt.title('SVM Decision Boundary')
    plt.xlabel('CompPrice')
    plt.ylabel('Income')
    plt.legend()

    plt.subplot(1, 2, 2)
    # Comparaison des prédictions avec les vraies valeurs

    plt.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], color='cyan', marker='o', label='Prédit High = 1',
                edgecolor='black', s=100, alpha=0.5)
    plt.scatter(X[y_pred == 0][:, 0], X[y_pred == 0][:, 1], color='orange', marker='x', label='Prédit High = 0',
                edgecolor='black', s=100, alpha=0.5)

    # Titre et étiquettes
    plt.title('Vraies Valeurs vs Prédictions')
    plt.xlabel('CompPrice')
    plt.ylabel('Income')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Appeler la fonction pour tracer l'hyperplan et la comparaison
plot_decision_boundary(X, y, model, y_pred)
