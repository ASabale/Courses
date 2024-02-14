import numpy as np
from sklearn.metrics import accuracy_score


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.n_iterations = max_epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iterations):
            # Calculate the predicted values
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)

            # Compute the gradients
            dw = np.dot(X.T, (y_pred - y)) / len(y)
            db = np.sum(y_pred - y) / len(y)

            # Update parameters using gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Make predictions
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        # Convert probabilities to binary predictions (0 or 1)
        return (y_pred >= 0.5).astype(int)

    def score(self, X, y):
        # Evaluate the accuracy of the model
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def save(self, file_path):
        """Save the model parameters to a file.

        Parameters:
        -----------
        file_path: str
            The file path to save the model parameters.
        """
        np.savez(file_path, weights=self.weights, bias=self.bias)

    def load(self, file_path):
        """Load the model parameters from a file.

        Parameters:
        -----------
        file_path: str
            The file path to load the model parameters.
        """
        data = np.load(file_path)
        self.weights = data['weights']
        self.bias = data['bias']