#           ***
# Linear models for CLASSIFICATION
#           ***

"""

Example of a binary classification :
y = w0 x0 + w1 x1 + w2 x2 + ... + wn xn + b > 0

y is the prediction made by the model
w0...wn and b are parameters of the model
x0...xn are the features 

Training the model will be learning new w0...wn and b parameters
A binary classifier, as defined, will be a line, a plane or an hyperplane
separating two classes.

"""

#           ***
# Logistic Regression and Linear Support Vector Classifier (SVC)
#           ***

import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2)
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()
plt.show()

# linear_models_classification_logisticRegression_vs_SVC.png

"""
Both SVC and Logitic regression use L2 like regularization.

This time, the parameter is called 'c'. The higher the 'c', the lower the regulation.

"""

mglearn.plots.plot_linear_svc_regularization()
plt.show()

# cf linear_svc_regularization.png


"""
As for regression, linear models for classification get more
powerful for higher dimensions data set, and that's where we 
should take care with overfitting 
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

# Training set score: 0.953
# Test set score: 0.958
# close values for training and test : maybe underfitting, so we
# should try to reduce regularization (higher C)

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

# Training set score: 0.972
# Test set score: 0.965

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

# Training set score: 0.934
# Test set score: 0.930

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.show()

# cf regularization_with_c.png

"""
We can also force these classifiers to use L1 regulation, to reduce the number of
used features
"""

for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, penalty="l1", solver='liblinear').fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-5, 5)
plt.legend(loc=3)
plt.show()

# Training accuracy of l1 logreg with C=0.001: 0.91
# Test accuracy of l1 logreg with C=0.001: 0.92
# Training accuracy of l1 logreg with C=1.000: 0.96
# Test accuracy of l1 logreg with C=1.000: 0.96
# Training accuracy of l1 logreg with C=100.000: 0.99
# Test accuracy of l1 logreg with C=100.000: 0.98

# cf l1_regularization_for_linear_classifiers.png