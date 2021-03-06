"""
Linear models use a linear function of the inputs features to make the prediction
"""

#           ***
# Linear model for REGRESSION
#           ***

"""
y = w0 x0 + w1 x1 + w2 x2 + ... + wn xn + b

y is the prediction made by the model
w0...wn and b are parameters of the model
x0...xn are the features 

Training the model will be learning new w0...wn and b parameters

"""

#           ***
# Linear REGRESSION (ordinary least squares)
#           ***

"""
OLS is the simplest and most classic method.

This linear regression tries to choose w and b, to minimize the mean
squared error between predictions and the true regression targets.
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mglearn
import numpy as np
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)

line = np.linspace(-3, 3, 100).reshape(-1, 1)
plt.figure(figsize=(8, 8))
plt.plot(line, lr.predict(line))
plt.plot(X, y, 'o')
ax = plt.gca()
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('center')
ax.spines['top'].set_color('none')

# cf linear-regression-OLS.png

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
# lr.coef_: [0.39390555]
# lr.intercept_: -0.031804343026759746
# Training set score: 0.67
# Test set score: 0.66

"""
Test set score is not good, but close to the training set score, so we are probably underfitting.

Linear models get better with more complex datasets.
Let's demonstrate it.
"""

X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print("Training set score : {}".format(lr.score(X_train, y_train)))
print("Test set score : {}".format(lr.score(X_test, y_test)))
# Training set score : 0.9520519609032728
# Test set score : 0.607472195966589

"""
In that case, we are obviously overfitting... Without changing the algorithm, so
it is only due to the complexity of the data set.

So the big issue is OLS, is that we don't have any control on complexity
"""

#           ***
# Ridge regularization
#           ***

"""
Ridge regression adds a 'penalty' based on the coefficient w and tries to 
minimize them (so that a single feature cannot have a too large impact)

https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c

example of adjustment :
"""

from sklearn.linear_model import Ridge
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))
# Training set score: 0.79
# Test set score: 0.64 

from sklearn.linear_model import Ridge
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))
# Training set score: 0.93
# Test set score: 0.77

"""
The Model will be more restricted with a higher alpha, so we should have
higher coeffs w with a lower alpha :
cf ridge_coeff_evolution_with_alpha.png
"""
ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()

#           ***
# Lasso regularization
#           ***

"""
Lasso, or L1 regularization, also adds a penalty based on the coefficient w,
but it is based on the sum of their Absolute values, and not their square anymore.

The consequence is that with Lasso, some coeff w will be exactly 0
"""

from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

# Training set score: 0.29
# Test set score: 0.21
# Number of features used: 4

"""
poor perf on both training and test set, it is underfitting.
Indeed it only used 4 features because most coeff w were at 0.
We should lower alpha, that is defaulted at 1.0
We should also increase 'max_iter' which the maximum number of
training iterations.
"""

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))
# Training set score: 0.90
# Test set score: 0.77
# Number of features used: 33

# if we set alpha too low, we fall back to the Linear Regression issue and do not benefit of regularization :
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))
# Training set score: 0.95
# Test set score: 0.64
# Number of features used: 96


# let's plot the coefficients w for diverse alpha :
plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
# cf cf lasso_coeff_evolution_with_alpha.png

"""
In practice, ridge regression is usually the first choice between these two models.
However, if you have a large amount of features and expect only a few of them to be
important, Lasso might be a better choice

PS : scikit-learn also provides the ElasticNet class that combines both L1 and L2 penalties.
It usually works the best but may be harder to adjust.
"""