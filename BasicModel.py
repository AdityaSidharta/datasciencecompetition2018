import autograd.numpy as np


class BasicModel:

    def _loss(self, A, B, bin_values, ground_truth):
        k = np.arange(len(bin_values))
        return np.mean(np.square(np.dot(5./60.,np.dot(A, np.matmul(bin_values, np.exp(k)**B))) - ground_truth))

    def fit(self, x, y, A=3, B=0.34, theta=0.0001, learning_rate=0.001):
        loss = []

        for i in range(len(x)):
            new = _loss(A, B, x[i], y[i])
            if loss:
                prev = loss[-1]
                if (max(new, prev) - min(new,prev)) < theta:
                    break
            loss.append(new)
            gradloss_a = grad(_loss, argnum=0)
            gradloss_b = grad(_loss, argnum=1)
            A = A - learning_rate*gradloss_a
            B = B - learning_rate*gradloss_b

        print("A: {}".format(self.A.weights))
        print("B: {}".format(self.B.weights))
        print("Loss: {}".format(loss[-1]))

    def predict(self, x):
        if not p:
            raise ValueError("Model is not fitted")
        A = self.A.weights
        B = self.B.weights
        y = A*np.exp(B*x)
        return y
