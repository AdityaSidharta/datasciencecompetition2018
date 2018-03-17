import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt


class BasicModel:
    def __init__(self):
        self.A = 3.
        self.B = .34
        self.X = None
        self.y = None

    def get_bin_size(self, X):
        if X.ndim == 1:
            assert len(X) == 34
            k = np.arange(len(X))
        else:
            assert X.shape[1] == 34
            k = np.arange(X.shape[1])
        return k

    def loss(self, A, B, X, y):
        k = self.get_bin_size(X)
        prediction = np.dot(1. / 12. * A, np.matmul(X, np.exp(k)**B))
        loss = np.mean(np.square(prediction - y))
        return loss

    def fit(self, X, y, A=None, B=None, learning_rate=0.000000001, max_iter=10000, theta=0.00000001):
        loss = []
        A = self.A if A is None else A
        B = self.B if B is None else B
        self.X = X
        self.y = y
        initial_loss = self.loss(A, B, X, y)
        print("Initial_loss: {}".format(initial_loss))
        print("Shape of x: {}".format(self.X.shape))
        print("Shape of y: {}".format(self.y.shape))

        for i in range(max_iter):
            new = self.loss(A, B, X, y)
            if loss:
                prev = loss[-1]
                if np.absolute(prev - new) / prev < theta:
                    print ("Fak ye bebi wi converge")
                    break
            loss.append(new)
            gradloss_a = grad(self.loss, argnum=0)
            gradloss_b = grad(self.loss, argnum=1)
            grad_a = gradloss_a(A, B, X, y)
            grad_b = gradloss_b(A, B, X, y)
            A = A - learning_rate * grad_a
            B = B - learning_rate * grad_b

        self.A = A
        self.B = B

        print("A: {}".format(self.A))
        print("B: {}".format(self.B))
        print("Loss: {}".format(loss[-1]))
        self.loss = loss

    def predict(self, X):
        if not (self.A and self.B):
            raise ValueError("Model is not fitted")
        else:
            A = self.A
            B = self.B
            k = self.get_bin_size(X)
            return np.dot(1. / 12. * A, np.matmul(X, np.exp(k) ** B))

    def score(self, X=None, y=None):
        if not (X and y):
            X = self.X
            y = self.y
        return self.loss(self.A, self.B, X, y)

    def plot_lost(self):
        plt.plot(np.arange(len(loss)), loss)
        plt.show()
