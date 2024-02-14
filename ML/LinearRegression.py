from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.learning_rate = None
        self.loss_history = []
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0.0, max_epochs=100, patience=3, learning_rate=0.01):
        """Fit a linear model.

        Parameters:
        -----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target values.
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        learning_rate: float
            The learning rate for gradient descent.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate

        # Initialize the weights and bias based on the shape of X and y.
        self.weights = np.random.randn(X.shape[1])
        self.bias = np.random.randn()

        X_training_set, X_validation_set, y_training_set, y_validation_set = train_test_split(X, y,
                                                                                              test_size=0.1,
                                                                                              random_state=20,
                                                                                              shuffle=False)

        weights = self.weights.copy()
        bias = self.bias
        best_val_loss = float('inf')
        counter = 0

        for epoch in range(self.max_epochs):
            for i in range(0, len(X_training_set), self.batch_size):
                X_batch = X_training_set[i:i + self.batch_size]
                y_batch = y_training_set[i:i + self.batch_size]
                predictions = np.dot(X_batch, self.weights) + self.bias

                error = predictions - y_batch
                gradient_weights = (2 / len(X_batch)) * np.dot(X_batch.T, error)
                gradient_bias = (2 / len(X_batch)) * np.sum(error)
                gradient_weights += 2 * self.regularization * self.weights
                self.weights -= learning_rate * gradient_weights
                self.bias -= learning_rate * gradient_bias

            val_predictions = self.predict(X_validation_set)
            val_loss = np.mean((val_predictions - y_validation_set) ** 2)
            self.loss_history.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                weights = self.weights.copy()
                bias = self.bias
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        # Set the model parameters to the best values.
        self.weights = weights
        self.bias = bias

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse

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


def modelSave(X, y, modelNum, title):
    X_train_scaled, X_test_scaled, y_train, y_test = createSets(X, y)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    filepath = "model-" + modelNum
    model.save(filepath + ".npz")
    model.load(filepath + ".npz")
    plt.plot(model.loss_history)
    plt.xlabel('Step Number')
    plt.ylabel('Mean Squared Error')
    plt.title("model-" + modelNum + "-" + title)
    plt.savefig("model-" + modelNum + '_loss_plot.png')
    plt.show()


def createSets(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    iris = load_iris()
    modelSave(iris.data[:, [0, 1]], iris.data[:, 2], "one", "sepal length against petal length")
    modelSave(iris.data[:, [1, 2]], iris.data[:, 0], "two", "sepal width and petal length against sepal length")
    modelSave(iris.data[:, [0, 2, 3]], iris.data[:, 1], "three",
              "sepal length, petal length and petal width against sepal width")
    modelSave(iris.data[:, [1]], iris.data[:, 3], "four", "sepal width against petal width")


if __name__ == '__main__':
    main()
