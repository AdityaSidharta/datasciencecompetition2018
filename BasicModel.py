import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from tqdm import tqdm

class BasicModel:
    def __init__(self):
        self.A = 0.079
        self.B = 0.228
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
        
    def fit(self, X, y, learning_rate=0.0001, beta_1 = 0.90, beta_2 = 0.999, max_iter=10000, theta=0.00000001, epsilon = 10 ** (-8)):
        loss = []
        A = self.A
        B = self.B
        self.X = X
        self.y = y
        initial_loss = self.loss(A, B, X, y)
        print("Initial_loss: {}".format(initial_loss))
        print("Shape of x: {}".format(self.X.shape))
        print("Shape of y: {}".format(self.y.shape))
        gradloss_a = grad(self.loss, argnum=0)
        gradloss_b = grad(self.loss, argnum=1)
        v_grad_a = 0
        v_grad_b = 0
        s_grad_a = 0
        s_grad_b = 0
        for i in tqdm(range(1,max_iter)):
            new = self.loss(A, B, X, y)
            loss.append(new)
            grad_a = gradloss_a(A, B, X, y)
            grad_b = gradloss_b(A, B, X, y)

            v_grad_a = (beta_1 * v_grad_a) + ((1 - beta_1) * grad_a)
            v_grad_b = (beta_1 * v_grad_b) + ((1 - beta_1) * grad_b)
            s_grad_a = (beta_2 * s_grad_a) + ((1 - beta_2) * grad_a**2)
            s_grad_b = (beta_2 * s_grad_b) + ((1 - beta_2) * grad_b**2)
            #v_grad_a = v_grad_a / (1 - beta_1**(i))
            #v_grad_b = v_grad_b / (1 - beta_1**(i))
            #s_grad_a = s_grad_a / (1 - beta_2**(i))
            #s_grad_b = s_grad_b / (1 - beta_2**(i))
            
            A = A - learning_rate * v_grad_a / np.sqrt(s_grad_a + epsilon)
            B = B - learning_rate * v_grad_b / np.sqrt(s_grad_b + epsilon)
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
        plt.plot(np.arange(len(self.loss)), self.loss)
        plt.show()
    
