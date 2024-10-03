from supportvectormachine import SupportVectorMachine
import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("../data/Carseats.csv")
    cible = df['High'].apply(lambda x: 1 if x == 'Yes' else -1).tolist()  # Utilisation de -1 pour correspondre à SVM
    df = df.drop(columns=['High'])
    list_feature = [df['CompPrice'].tolist(), df['Income'].tolist(), df['Advertising'].tolist(),df['Population'].tolist(), df['Price'].tolist(), df['Age'].tolist()]
    SVM = SupportVectorMachine(list_feature, cible)
    SVM.fit()
    predictions = SVM.predict(list_feature)
    SVM.accuracy()

