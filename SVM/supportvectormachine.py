class SupportVectorMachine:
    def __init__(self, df, df_cible):
        self.df = df  # df est une liste de listes, où chaque sous-liste est une colonne
        self.df_cible = df_cible
        self.poids = [0] * len(df[0])  # initialisation des poids pour chaque feature (colonne)
        self.biais = 0  # initialisation du biais à 0

    def predict(self, X):
        # Calculer le produit scalaire et ajouter le biais
        return [1 if (sum(x * w for x, w in zip(X_i, self.poids)) + self.biais) >= 0 else -1 for X_i in zip(*X)]

    def fit(self):
        learning_rate = 0.01
        regularization_parameter = 0.01

        for nbdessaie in range(1000):  # nombre d'itérations
            for i in range(len(self.df_cible)):
                x_i = [self.df[j][i] for j in range(len(self.df))]  # obtenir la i-ème ligne
                y_i = self.df_cible[i]
                sum_scalaire = sum(x * w for x, w in zip(x_i, self.poids))
                print(len(self.poids))
                print(len(x_i))
                # Vérification de la condition du SVM
                if (sum_scalaire - self.biais) * y_i < 1:  # erreur de classification
                    # Mettre à jour les poids et le biais
                    for j in range(len(self.poids)):
                        self.poids[j] -= learning_rate * (regularization_parameter * self.poids[j] - y_i * x_i[j])
                    self.biais -= learning_rate * (-y_i)
                else:
                    # Si classé correctement, juste faire une petite régularisation
                    for j in range(len(self.poids)):
                        self.poids[j] -= learning_rate * regularization_parameter * self.poids[j]

    def accuracy(self):
        predictions = self.predict(self.df)  # Obtenir les prédictions sur l'ensemble de données
        correct_predictions = sum(1 for pred, true in zip(predictions, self.df_cible) if pred == true)
        accuracy = correct_predictions / len(self.df_cible)  # Calculer l'accuracy
        print(f'Accuracy: {accuracy * 100:.2f}%')
