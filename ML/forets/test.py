from classeRdForest import Foret
import numpy as np
import pandas as pds
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


df = pds.read_csv("C:\\Users\\ninop\\Documents\\ML-isen-grp4\\data\\Carseats.csv")

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(100)

X = df.drop(['Unnamed: 0','High'],axis= 1)
X = pds.get_dummies(X)
y = df['High']

# foret = Foret()
# foret.fit(X,y)
# y_pred = foret.predict(X)

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
y_pred = clf.predict(X)

print(accuracy_score(y, y_pred))




