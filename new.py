import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ---------------- Linear regression example ----------------
height = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
weight = [8, 10, 12, 14, 16, 18, 20]

plt.scatter(height, weight, color="black")
plt.xlabel("height")
plt.ylabel("weight")

reg = linear_model.LinearRegression()
reg.fit(height, weight)

X_height = [[12.0]]
print("Predicted weight:", reg.predict(X_height))


# ---------------- Train-test split example ----------------
X = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
y = [8, 10, 12, 14, 16, 18, 20]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7
)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

result = reg.score(X_test, y_test)
print("Accuracy - test set: %.2f%%" % (result * 100.0))


# ---------------- Polynomial regression example ----------------
x = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
y = [16, 25, 36, 49, 64, 81, 100]

lin_reg = LinearRegression()
lin_reg.fit(x, y)

print("Linear prediction for 11:", lin_reg.predict([[11]]))

polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression(),
)

polynomial_regression.fit(x, y)

target_predicted = polynomial_regression.predict([[20.0]])
print("Polynomial prediction for 20:", target_predicted)
