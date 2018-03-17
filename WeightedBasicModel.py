import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from tqdm import tqdm


class WeightedBasicModel:
    def __init__(self):
        self.A = 0.079
        self.B = 0.228
        self.X = None
        self.y = None

    def get_bin_size(self, X):
        if X.ndim == 1:
            assert len(X) == 37
            k = np.arange(len(X) - 3)
        else:
            assert X.shape[1] == 37
            k = np.arange(X.shape[1] - 3)
        return k

    def loss(self, A, B, X, y, prob):
        ''' y is a matrix of size len(X), 50'''
        assert y.ndim == 2
        assert y.shape[1] == 50

        k = self.get_bin_size(X)
        prediction = np.dot(1. / 12. * A, np.matmul(X[[str(i) for i in range(34)]].values, np.exp(B * k)))
        loss = np.mean(np.multiply(np.square(prediction - y.values.T), prob.T))
        return loss

    def fit(self, X, y, learning_rate=0.0001, beta_1=0.90, beta_2=0.999, max_iter=10000, theta=0.00000001,
            epsilon=10 ** (-8), prob_table=None):

        loss = []
        A = self.A
        B = self.B
        self.X = X
        self.y = y
        self.prob_table = prob_table
        idx = map(lambda x, y: (x, y), X["lat"], X["long"])
        prob = self.prob_table.set_index(["lat", "long"]).loc[idx].values
        initial_loss = self.loss(A, B, X, y, prob)
        print("Initial_loss: {}".format(initial_loss))
        print("Shape of x: {}".format(self.X.shape))
        print("Shape of y: {}".format(self.y.shape))
        gradloss_a = grad(self.loss, argnum=0)
        gradloss_b = grad(self.loss, argnum=1)
        v_grad_a = 0
        v_grad_b = 0
        s_grad_a = 0
        s_grad_b = 0
        for i in tqdm(range(1, max_iter)):
            new = self.loss(A, B, X, y, prob)
            loss.append(new)
            grad_a = gradloss_a(A, B, X, y, prob)
            grad_b = gradloss_b(A, B, X, y, prob)

            v_grad_a = (beta_1 * v_grad_a) + ((1 - beta_1) * grad_a)
            v_grad_b = (beta_1 * v_grad_b) + ((1 - beta_1) * grad_b)
            s_grad_a = (beta_2 * s_grad_a) + ((1 - beta_2) * grad_a**2)
            s_grad_b = (beta_2 * s_grad_b) + ((1 - beta_2) * grad_b**2)

            A = A - learning_rate * v_grad_a / np.sqrt(s_grad_a + epsilon)
            B = B - learning_rate * v_grad_b / np.sqrt(s_grad_b + epsilon)
        self.A = A
        self.B = B

        print("A: {}".format(self.A))
        print("B: {}".format(self.B))
        print("Loss: {}".format(loss[-1]))
        self.loss = loss
        return self

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
        plt.plot(np.arange(len(self.loss)), self.loss)
        plt.show()


if __name__ == '__main__':
    import pandas as pd
    X = pd.DataFrame(np.random.randn(34000).reshape(1000, 34))
    y = np.random.randn(1000 * 50).reshape(1000, 50)
    X["lat"] = np.arange(1000)
    X["long"] = np.arange(1000)
    prob_table = pd.DataFrame(np.random.randn(1000 * 50).reshape(1000, 50))
    prob_table["lat"] = X.lat
    prob_table["long"] = X.long
    X = X[["lat", "long"] + [i for i in range(34)]]
    clf = WeightedBasicModel().fit(X, y, prob_table=prob_table)
    clf.plot_lost()
