class LogisticRegression:
    def __init__(self, alpha = 0.01, epoch = 15000):
        self.alpha = alpha
        self.epoch = epoch

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.costs   = []

        for _ in range(self.epoch):
            output = self.sigmoid(self.net_input(X))
            errors = y - output
            self.weights[1:] += self.alpha * X.T.dot(errors)
            self.weights[0]  += self.alpha * errors.sum()
            cost = self.cost(X, y)
            self.costs.append(cost)
        return self

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    def cost(self, X, y):
        net_input = self.net_input(X)
        cost = -np.sum(y * np.log(self.sigmoid(net_input))
                + (1 - y) * np.log(1 - self.sigmoid(net_input)))
        return cost

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return self.sigmoid(self.net_input(X))