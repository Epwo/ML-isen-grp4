from supportvectormachine import SupportVectorMachine
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("../data/Carseats.csv")
    SVM = SupportVectorMachine(df)
    print(SVM)
