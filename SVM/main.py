from supportvectormachine import SupportVectorMachine
import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("../data/Carseats.csv")
    l = np.random.randint(2, size=400)
    l1 = np.random.randint(2, size=400)
    l2 = np.random.randint(2, size=400)
    l3 = np.random.randint(2, size=400)
    SVM = SupportVectorMachine([l1, l2, l3], l)
    SVM.extractData()
    print(SVM.extractData())
