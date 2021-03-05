"""
Also called k-NN algorithm

Training here only consist of storing the training set.
The prediction (classification) is made by finding the class of 
the nearest neighbors.
"""

 #      ***
 #  k-neighbor for CLASSIFICATION
 #      ***

import mglearn

mglearn.plots.plot_knn_classification(n_neighbors=1)

# example at 1-nearest-neighbor-example.png

mglearn.plots.plot_knn_classification(n_neighbors=3)

# example at 3-nearest-neighbor-example.png
# the "voting" process, picks the most frequent class of neighbors.

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set classes : {}".format(y_test))
# Test set predictions: [1 0 1 0 1 0 0]
# Test set classes : [1 0 1 0 1 1 0]
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))
# Test set accuracy: 0.86

# we can directly illustrate the "prediction bundaries" on a plane,
# since we only have 2 dimensions here.
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)

# example at prediction_boundaries_example.png
"""
We can observe that the model gets smoother (simpler) with more neighbor.
"""

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

# check k-neighbours_accuracy_comparison.png
"""
We can now observe the impact of over and under-fitting
"""

 #      ***
 #  k-neighbor for REGRESSION
 #      ***

# This gives us an example of a k-neighbor regression approach :
mglearn.plots.plot_knn_regression(n_neighbors=1)
"""
k-neighbors_regression_example.png

Prediction will simply be the target value of the nearest neighbor
"""

mglearn.plots.plot_knn_regression(n_neighbors=3)
"""
We can smooth the model using more neighbor : 3-neighbors_regression_example.png
"""

from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)
# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
reg.fit(X_train, y_train)
print("Test set predictions:\n{}".format(reg.predict(X_test)))
print("Test set actual values:\n {}".format(y_test))
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

"""
Coefficient of determination :
Rˆ2 is a score to determine the accuracy of a regression algorithm.
1 is a perfect prediction
0 is just the mean of the training set responses
"""
import numpy as np 

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1,000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # make predictions using 1, 3, or 9 neighbors
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target","Test data/target"], loc="best")

"""
check 1/2/3-k_neighbor_regression_example.png

Using multiple neighbors obviously reduce the influence of single points

There are 2 parameters to adjust :
- number of neighbors
- distance measure (we only use Euclidian distance for now with Rˆ2)
"""

#           ***
#   ADVANTAGES
#           ***

# Easy to understand
# Reasonable performances without too many adjustments

#           ***
#   DISADVANTAGES
#           ***

# Can be pretty slow if the training model has a lot of datas or features