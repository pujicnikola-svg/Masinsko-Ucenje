# Zadatak 2.3 - kratak test skript za klasu PolynomialRegression
# Nikola Pujic 2519/26

import pandas as pd
from polinomijalna_regresija import PolynomialRegression

# ucitavanje podataka
df = pd.read_csv("podaci.csv")

X = df["X"].values
y = df["Y"].values

M = 2
model = PolynomialRegression(degree=M)
model.fit(X, y)

print("\nStepen polinoma M =", M)
print("Parametri w:", model.coef_)

y_pred = model.predict(X)
print("\nX      Y      Y_pred")
for i in range(len(X)):
    print(X[i], " ", y[i], " ", round(y_pred[i], 3))

print("\nMSE na skupu za obuku:", model.score(X, y))
