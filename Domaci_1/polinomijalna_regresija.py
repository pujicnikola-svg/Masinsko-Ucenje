# Zadatak 2 - Polinomijalna regresija kompatibilna sa scikit-learn
# Nikola Pujic 2519/26

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class PolynomialRegression(BaseEstimator, RegressorMixin):

    def __init__(self, degree=1):
        # degree = stepen polinoma M
        self.degree = degree

    def _napravi_phi(self, X):
        # pretvaramo X u 1D niz (jedna ulazna promenljiva)
        x = np.asarray(X, dtype=float).reshape(-1)
        # Phi[i, j] = x_i^j, j = 0..M  (prva kolona su jedinice)
        Phi = np.vander(x, self.degree + 1, increasing=True)
        return Phi

    def fit(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        Phi = self._napravi_phi(X)

        # sistem normalnih jednacina: (Phi^T Phi) w = Phi^T y
        A = Phi.T @ Phi
        b = Phi.T @ y

        # probamo prvo solve, ako je matrica singularna koristimo lstsq
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("Upozorenje: singularna matrica, koristim lstsq")
            w = np.linalg.lstsq(A, b, rcond=None)[0]

        self.coef_ = w
        return self

    def predict(self, X):
        Phi = self._napravi_phi(X)
        y_pred = Phi @ self.coef_
        return y_pred

    def score(self, X, y):
        # MSE = 1/N * suma (y_i - y_pred_i)^2
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse

    def error(self, X, y):
        # funkcija greske iz zad. 1: Err(w) = 1/(2N) * suma(...)^2 = MSE/2
        return self.score(X, y) / 2
