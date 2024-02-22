import numpy as np


class LDAModel:
    def __init__(self):
        self.shared_covariance = None
        self.class_covariances = None
        self.class_means = None

    def fit(self, X, y):
        classes = np.unique(y)
        class_means = []
        class_covariances = []

        for c in classes:
            class_data = X[y == c]

            mean = np.mean(class_data, axis=0)
            class_means.append(mean)

            covariance = np.cov(class_data, rowvar=False)
            class_covariances.append(covariance)

        # Convert to numpy arrays for easier manipulation
        self.class_means = np.array(class_means)
        self.class_covariances = np.array(class_covariances)

        # Calculate the shared covariance matrix
        self.shared_covariance = np.mean(self.class_covariances, axis=0)

    def predict(self, X):
        # Calculate the discriminant function for each class
        discriminants = []
        for i in range(len(self.class_means)):
            mean_diff = X - self.class_means[i]
            inv_covariance = np.linalg.inv(self.shared_covariance)
            discriminant = -0.5 * np.sum(mean_diff @ inv_covariance * mean_diff, axis=1)
            discriminants.append(discriminant)

        # Choose the class with the highest discriminant value
        predictions = np.argmax(np.array(discriminants), axis=0)

        return predictions


class QDAModel:
    def fit(self, X, y):
        # TODO: Implement the fit method
        pass

    def predict(self, X):
        # TODO: Implement the predict method
        pass


class GaussianNBModel:
    def fit(self, X, y):
        # TODO: Implement the fit method
        pass

    def predict(self, X):
        # TODO: Implement the predict method
        pass
