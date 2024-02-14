from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(learning_rate=0.01, max_epochs=1000)
model.fit(X_train_scaled, y_train)

filepath = "LogisticRegression Model-1"
model.save(filepath + ".npz")
model.load(filepath + ".npz")
y_pred = model.predict(X_test_scaled)

plot_decision_regions(X_train_scaled, y_train, clf=model, legend=2)
plt.title('Logistic Regression - Petal Length/Width')
plt.xlabel('Petal Length (standardized)')
plt.ylabel('Petal Width (standardized)')
filename = "LogisticRegression 1.png"
plt.savefig(filename)
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
