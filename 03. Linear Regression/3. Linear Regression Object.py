import numpy as np

class LinearRegression:
    def __init__(self, eta = 0.01, epoch = 15000):
        self.eta   = eta
        self.epoch = epoch

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.costs   = []

        for _ in range(self.epoch):
            output = self.net_input(X)
            errors = y - output
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0]  += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.costs.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return self.net_input(X)

