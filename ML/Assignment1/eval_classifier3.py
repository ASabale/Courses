from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20, shuffle=False)
model = LogisticRegression(learning_rate=0.01, max_epochs=1000)
model.fit(X_train, y_train)

filepath = "LogisticRegression Model-3"
model.save(filepath + ".npz")
model.load(filepath + ".npz")

predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
