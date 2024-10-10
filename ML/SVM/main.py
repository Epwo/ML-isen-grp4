from supportvectormachine import SupportVectorMachine, z_score_normalize, train_test_split_custom, affichage
import numpy as np
import pandas as pd





if __name__ == "__main__":
    # traitement des données
    df = pd.read_csv('../../data/Carseats.csv')
    df['ShelveLoc'] = df['ShelveLoc'].map({'Bad': 0, 'Medium': 1, 'Good': 2})
    df['Urban'] = df['Urban'].map({'No': 0, 'Yes': 1})
    df['US'] = df['US'].map({'No': 0, 'Yes': 1})
    df['High'] = df['High'].map({'No': 0, 'Yes': 1})
    # isolement donnée cible
    X = df.drop('High', axis=1).values
    y = df['High'].values
    # uniformement notamment pour les variables qualitatives
    X_normalized = z_score_normalize(X)
    # Séparation des données
    X_train, X_test, y_train, y_test = train_test_split_custom(X_normalized, y, test_size=0.2)
    # Initialiser et entraîner le modèle SVM
    svm = SupportVectorMachine(learning_rate=0.005, lambda_param=0.01, n_iters=30000)
    svm.fit(X_train, y_train)
    # prédiction + performance
    y_pred = svm.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")
    affichage(y_test, y_pred)
