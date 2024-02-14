import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

iris = load_iris()
X, y = iris.data[:, [1, 2]], iris.data[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20, shuffle=False)
model = LinearRegression()
filepath = "model-two"
model.load(filepath + ".npz")
model_reg = LinearRegression(batch_size=32, regularization=0.02, max_epochs=100, patience=3)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model_reg.fit(X_train_scaled, y_train)

plt.plot(model_reg.loss_history)
plt.xlabel('Step Number')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss - model 2 with Regularization')
plt.savefig('model-two_with_regularization_loss_plot.png')
plt.show()

# Record the difference in parameters
difference_in_weights = np.sum(np.abs(model_reg.weights - model.weights))
difference_in_bias = np.sum(np.abs(model_reg.bias - model.bias))

print("Difference in weights between regularized and non-regularized model:", difference_in_weights)
print("Difference in bias between regularized and non-regularized model:", difference_in_bias)
