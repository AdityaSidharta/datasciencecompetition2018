import autograd.numpy as np
from autograd import grad


class BasicModel:

    def __init__(self):
        self.A = None
        self.B = None
        self.X = None
        self.y = None

    def _loss(self, A, B, bin_values, ground_truth):
        if bin_values.ndim == 1:
            k = np.arange(len(bin_values))
        else:
            k = np.arange(bin_values.shape[1])
        return np.mean(np.square(np.dot(1 / 12. * A, np.matmul(bin_values, np.exp(k)**B)) - ground_truth))

    def fit(self, X, y, A=3., B=0.34, theta=0.0001, learning_rate=0.001, max_iter=10000):
        loss = []
        self.X = X
        self.y = y

        for i in range(max_iter):
            new = self._loss(A, B, X, y)
            if loss:
                prev = loss[-1]
                if np.absolute(prev - new) / prev < theta:
                    break
            loss.append(new)
            gradloss_a = grad(self._loss, argnum=0)(A, B, X, y)
            gradloss_b = grad(self._loss, argnum=1)(A, B, X, y)
            A = A - learning_rate * gradloss_a
            B = B - learning_rate * gradloss_b

        self.A = A
        self.B = B

        print("A: {}".format(self.A))
        print("B: {}".format(self.B))
        print("Loss: {}".format(loss[-1]))
        return self

    def predict(self, X):
        if not (self.A and self.B):
            raise ValueError("Model is not fitted")
        else:
            A = self.A
            B = self.B
            if X.ndim == 1:
                k = np.arange(len(X))
            else:
                k = np.arange(X.shape[1])
            return np.dot(1 / 12. * A, np.matmul(X, np.exp(k) ** B))

    def score(self, X=None, y=None):
        if not (X and y):
            X = self.X
            y = self.y
        return self._loss(self.A, self.B, X, y)


if __name__ == '__main__':
    X = np.random.randn(1000).reshape(200, 5)
    y = np.random.randn(200)
    clf = BasicModel().fit(X, y)
    print clf.score()
