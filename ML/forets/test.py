from classeRdForest import Foret
import numpy as np

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(100)

foret = Foret()
foret.fit(X,y)

y_pred = foret.predict(X)

print(y_pred)




