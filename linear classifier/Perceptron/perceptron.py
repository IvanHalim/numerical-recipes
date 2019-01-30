import numpy as np

## Perceptrons are the building blocks of neural networks
## they are the basic units from which more complex neural networks are built.

class Perceptron:
    def __init__(self, alpha = 0.01, epoch = 10):
        ## alpha is the learning rate
        ## epoch is the number of passes over the training set
        self.alpha = alpha
        self.epoch = epoch

    def fit(self, X, y):
        ## self.weights is a 1d array with a length of 1 + the number of features in X.
        ## We add 1 for the zero-weight, which is the bias/threshold
        self.weights = np.zeros(1 + X.shape[1])

        # We want to find how many times the perceptron predicted wrongly in each epoch
        # self.errors is a list where self.errors[0] is the number of errors in epoch 1,
        # self.errors[1] is the number of errors in epoch 2, etc
        self.errors  = []

        for _ in range(self.epoch):
            error = 0 # Initialize a zero error
            for xi, target in zip(X, y):

                ## We calculate the update by finding the difference between
                ## the actual label and the predicted label, multiplied by the
                ## learning rate.
                update = self.alpha * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0]  += update

                # Increment the error if update is not zero
                error += int(update != 0.0)

            # At the end of each epoch, append error to self.errors
            self.errors.append(error)
        return self

    def net_input(self, X):
        ## Net input is weighted sum, that is the sum of w*x plus the bias
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        ## This is the activation function, which uses the Heaviside step function
        ## If net input > 0.0 then return 1, else return -1
        return np.where(self.net_input(X) >= 0.0, 1, -1)
