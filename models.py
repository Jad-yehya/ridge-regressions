
import numpy as np
from tqdm import tqdm

class RidgeRegression:
    """
    Ridge regression model with closed form solution
    """
    def __init__(self, lambda_, shape, rand_init=True):
        self.lambda_ = lambda_
        # Adding 1 dimension for the bias term
        if rand_init:
            self.w = np.random.randn(shape+1, 1)
        else:
            self.w = np.zeros((shape+1, 1))


    def train(self, X, y, n_epochs=1000):
        """
        X: matrix of shape (n_samples, n_features)
        y: vector of shape (n_samples, 1)
        n_epochs: scalar
        returns: vector of shape (n_features, 1)
        """
        for i in range(n_epochs):
            self.w = np.linalg.inv(X.T @ X + self.lambda_ * np.eye(X.shape[1])) @ X.T @ y
        return self.w


    def predict(self, X):
        """
        X: matrix of shape (n_samples, n_features)
        w: vector of shape (n_features, 1)
        returns: vector of shape (n_samples, 1)
        """
        return np.dot(X, self.w)

    def compute_error(self, y_true, y_pred):
        """
        y_true: vector of shape (n_samples, 1)
        y_pred: vector of shape (n_samples, 1)
        returns: scalar
        """
        return np.mean((y_true - y_pred)**2)

    def cross_validation(self, X, y, k=5):
        """
        Coding k-fold cross validation from scratch
        X: matrix of shape (n_samples, n_features)
        y: vector of shape (n_samples, 1)
        k: scalar
        returns: vector of shape (k, 1)
        """
        n_samples = X.shape[0]
        n_samples_fold = n_samples // k
        errors = []
        for i in range(k):
            X_train = np.concatenate((X[:i*n_samples_fold], X[(i+1)*n_samples_fold:]))
            y_train = np.concatenate((y[:i*n_samples_fold], y[(i+1)*n_samples_fold:]))
            X_test = X[i*n_samples_fold:(i+1)*n_samples_fold]
            y_test = y[i*n_samples_fold:(i+1)*n_samples_fold]
            self.train(X_train, y_train)
            y_pred = self.predict(X_test)
            errors.append(self.compute_error(y_test, y_pred))
        return errors



class RidgeRegressionSGD:
    """
    Ridge regression model with stochastic gradient descent
    """
    def __init__(self, lambda_, shape, rand_init=True):
        self.lambda_ = lambda_
        if rand_init:
            self.w = np.random.randn(shape, 1)
        else:
            self.w = np.zeros((shape, 1))


    def train(self, X, y, n_epochs=1000, learning_rate=0.01, epsilon=1e-3):
        """        
        X: matrix of shape (n_samples, n_features)
        y: vector of shape (n_samples, 1)
        n_epochs: scalar
        learning_rate: scalar
        epsilon: scalar, threshold for early stopping
        """
        n_samples, n_features = X.shape
        old_w = self.w.copy()  # Copy the initial weights

        for epoch in range(n_epochs):
            # Iterate over each sample in a random order
            indexes = np.random.permutation(n_samples)
            for idx in indexes:
                # Compute the prediction for the current sample
                x = X[idx, :]
                y_pred = np.dot(x, self.w)

                # Update the weights using stochastic gradient descent
                gradient = -2 * (y[idx] - y_pred) * x.reshape(-1, 1) + 2 * self.lambda_ * self.w
                self.w -= learning_rate * gradient

            # Regularize the weights except for the bias term
            self.w[1:] -= learning_rate * 2 * self.lambda_ * self.w[1:]

            # Check for early stopping
            if np.linalg.norm(old_w - self.w) < epsilon:
                print(f"Early stopping at epoch {epoch+1}")
                break

            old_w = self.w.copy()  # Update old weights for the next epoch

        return self.w


    def predict(self, X):
        """
        X: matrix of shape (n_samples, n_features)
        w: vector of shape (n_features, 1)
        returns: vector of shape (n_samples, 1)
        """
        return np.dot(X, self.w)

    def compute_error(self, y_true, y_pred):
        """
        y_true: vector of shape (n_samples, 1)
        y_pred: vector of shape (n_samples, 1)
        returns: scalar
        """
        return np.mean((y_true - y_pred)**2)

    def cross_validation(self, X, y, n_epochs, k=5):
        """
        Coding k-fold cross validation from scratch
        X: matrix of shape (n_samples, n_features)
        y: vector of shape (n_samples, 1)
        k: scalar
        returns: vector of shape (k, 1)
        """
        n_samples = X.shape[0]
        n_samples_fold = n_samples // k
        errors = []
        for i in range(k):
            X_train = np.concatenate((X[:i*n_samples_fold], X[(i+1)*n_samples_fold:]))
            y_train = np.concatenate((y[:i*n_samples_fold], y[(i+1)*n_samples_fold:]))
            X_test = X[i*n_samples_fold:(i+1)*n_samples_fold]
            y_test = y[i*n_samples_fold:(i+1)*n_samples_fold]
            self.train(X_train, y_train, n_epochs=n_epochs)
            y_pred = self.predict(X_test)
            errors.append(self.compute_error(y_test, y_pred))
        return errors
    

class GaussianKernelRidgeRegression:
    def __init__(self, lambda_, gamma, shape, rand_init=True):
        """
        lambda_: scalar
        gamma: scalar
        shape: scalar
        rand_init: boolean
        """
        self.lambda_ = lambda_
        self.gamma = gamma
        if rand_init:
            self.w = np.random.randn(shape, 1)
        else:
            self.w = np.zeros((shape, 1))


    def train(self, X, y):
        """
        X: matrix of shape (n_samples, n_features)
        y: vector of shape (n_samples, 1)
        """
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i in tqdm(range(n_samples)):
            for j in range(n_samples):
                K[i, j] = np.exp(-(1/(2*self.gamma)) * np.linalg.norm(X[i] - X[j])**2)
        self.w = np.linalg.inv(K + self.lambda_ * np.eye(n_samples)) @ y
        return self.w


    def predict(self, X):
        """
        X: matrix of shape (n_samples, n_features)
        w: vector of shape (n_features, 1)
        returns: vector of shape (n_samples, 1)
        """
        n_samples, n_features = X.shape
        y_pred = np.zeros((n_samples, 1))
        for i in range(n_samples):
            for j in range(n_samples):
                y_pred[i] += self.w[j] * np.exp(-(1/(2*self.gamma)) * np.linalg.norm(X[i] - X[j])**2)
        return y_pred

    def compute_error(self, y_true, y_pred):
        """
        y_true: vector of shape (n_samples, 1)
        y_pred: vector of shape (n_samples, 1)
        returns: scalar
        """
        return np.mean((y_true - y_pred)**2)

    def cross_validation(self, X, y, k=5):
        """
        Coding k-fold cross validation from scratch
        X: matrix of shape (n_samples, n_features)
        y: vector of shape (n_samples, 1)
        k: scalar
        returns: vector of shape (k, 1)
        """
        n_samples = X.shape[0]
        n_samples_fold = n_samples // k
        errors = []
        for i in (range(k)):
            X_train = np.concatenate((X[:i*n_samples_fold], X[(i+1)*n_samples_fold:]))
            y_train = np.concatenate((y[:i*n_samples_fold], y[(i+1)*n_samples_fold:]))
            X_test = X[i*n_samples_fold:(i+1)*n_samples_fold]
            y_test = y[i*n_samples_fold:(i+1)*n_samples_fold]
            self.train(X_train, y_train)
            y_pred = self.predict(X_test)
            errors.append(self.compute_error(y_test, y_pred))
        return errors
