import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class CustomSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=2000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialisation des poids et du biais
        self.w = np.zeros(n_features)
        self.b = 0
        y_ = np.where(y <= 0, -1, 1) 

        # Descente de gradient
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.where(approx >= 0, 1, 0) 

# SVM amélioré avec ajustement des paramètres et plus d'itérations
class FurtherImprovedSVM(CustomSVM):
    def __init__(self, learning_rate=0.005, lambda_param=0.01, n_iters=30000):
        super().__init__(learning_rate, lambda_param, n_iters)

# Normalisation Z-score
def z_score_normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Séparation améliorée des données
def train_test_split_custom(X, y, test_size=0.2):
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_idx, train_idx = indices[:test_size], indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Charger les données
df = pd.read_csv('C:\ISEN-DUCLOS\project machine learning\data\Carseats.csv')
df['ShelveLoc'] = df['ShelveLoc'].map({'Bad': 0, 'Medium': 1, 'Good': 2})
df['Urban'] = df['Urban'].map({'No': 0, 'Yes': 1})
df['US'] = df['US'].map({'No': 0, 'Yes': 1})
df['High'] = df['High'].map({'No': 0, 'Yes': 1})

# Features et cible
X = df.drop('High', axis=1).values
y = df['High'].values

# Normalisation Z-score
X_normalized = z_score_normalize(X)

# Séparer les données
X_train, X_test, y_train, y_test = train_test_split_custom(X_normalized, y, test_size=0.2)

# Initialiser et entraîner le modèle SVM amélioré
further_improved_svm = FurtherImprovedSVM(learning_rate=0.005, lambda_param=0.01, n_iters=30000)
further_improved_svm.fit(X_train, y_train)

# Faire des prédictions
y_pred_further_improved = further_improved_svm.predict(X_test)

# Calculer la précision
accuracy_further_improved = np.mean(y_pred_further_improved == y_test)
print(f"Further Improved Accuracy: {accuracy_further_improved}")

# Visualisation des prédictions
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Vrais labels', alpha=0.6)
plt.scatter(range(len(y_pred_further_improved)), y_pred_further_improved, color='red', label='Prédictions', alpha=0.6)
plt.title('Comparaison des labels réels et des prédictions après amélioration')
plt.xlabel('Index')
plt.ylabel('Label (0: Non, 1: Oui)')
plt.legend(loc='best')
plt.show()
