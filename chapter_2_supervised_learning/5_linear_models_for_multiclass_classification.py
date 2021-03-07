#           ***
# Linear models for MULTICLASS CLASSIFICATION
#           ***

"""
Most of the linear classification models are for binary classification only, but can be used
for a one-vs-rest approach.  All the models have to be run against a test point to classify
it.

So you have a list of coeff w and b, for each class.
"""

import matplotlib.pyplot as plt
import numpy as np
import mglearn
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])
plt.show()

# This is the shape of our dataset (we can observe 3 classes, and 2 features):
# linear_model_multiclass_classification_one-vs-rest.png

linear_svm = LinearSVC().fit(X, y)
print("Coefficient shape: ", linear_svm.coef_.shape)
print("Intercept shape: ", linear_svm.intercept_.shape)

# Coefficient shape: (3, 2)
# Intercept shape: (3,)

"""
coef_ : is (3,2) :
    So it has 3 row, and 2 columns.
    Each column holds the coeff for the 2 features.
    Each row holds the coeffs for the a given class.

intercept_ : is (3,)
    So it has 3 lines, and 1 column.
    Each row only holds the intercept_ b for each of the 3 classes.
"""

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
        ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
'Line class 2'], loc=(1.01, 0.3))
plt.show()

# cf linear_model_multiclass_classification_one-vs-rest_lines.png

mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
        ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
'Line class 2'], loc=(1.01, 0.3))
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

# We can nos display the classes "areas":
# cf linear_model_multiclass_classification_one-vs-rest_areas.png