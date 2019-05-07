import numpy as np

class LogisticRegression:
    def __init__(self, eta = 0.01, epoch = 15000):
        self.eta   = eta
        self.epoch = epoch

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.costs   = []

        for _ in range(self.epoch):
            output = self.sigmoid(self.net_input(X))
            errors = y - output
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0]  += self.eta * errors.sum()
            cost = self.cost(X, y)
            self.costs.append(cost)
        return self

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    def cost(self, X, y):
        net_input = self.net_input(X)
        net_input_pos = net_input[y == 1]
        net_input_neg = net_input[y == 0]
        cost = - (1 / X.shape[0]) \
                * (np.sum(np.log(self.sigmoid(net_input_pos) + 10**(-16))) 
                    + np.sum(np.log(1 - self.sigmoid(net_input_neg) + 10**(-16))))
        return cost

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X, threshold = 0.5):
        return self.sigmoid(self.net_input(X)) >= threshold