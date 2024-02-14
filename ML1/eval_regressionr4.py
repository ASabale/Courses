from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from LinearRegression import LinearRegression

iris = load_iris()
model = LinearRegression()
X, y = iris.data[:, 0:2], iris.data[:, 3]
model.load('model4.npz')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5, shuffle=False)
mse_test = model.score(X_test, y_test)
print("Mean Squared Error on Test Set (Model 4):", mse_test)
