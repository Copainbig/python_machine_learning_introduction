"""
The goal of this example, is to create a model, to classify Iris flowers
based on their measurements (length and width of petals, and length and 
width of sepals), using existing measurements of existing species (setosa,
versicolor, virginica).

So it is a case of "supervised" machine learning.
We want to predict one out of several options (called "classes") : it is a
"classification" problem.
The option that will be given to an entity is called its "label"
"""



from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'] + "\n...")
print("Target names : {}".format(iris_dataset['target_names']))
print("Feature names : {}".format(iris_dataset['feature_names']))
# data contains a NumPy array with the measurements :
print("Type of data : {}".format(type(iris_dataset['data'])))
# rows are the individuals flowers, columns are the measurements
print("Shape of data : {}".format(iris_dataset['data'].shape))
# print first 5 columns of data
# each line is a "sample", and each column is a "feature"
print("First 5 columns of data : \n {}".format(iris_dataset['data'][:5]))
# target array contains the species of the flowers, as a NumPy array , encoded as integers, from 0 to 2.
# according to target_names :
# 0 => Setosa
# 1 => Versicolor
# 2 => Virginica
print("Type of flowers measured : \n {}".format(iris_dataset['target']))

"""
We want to build a machine learning model from this data set.
But we cannot evaluate the model, with the data we used to create it.
So we split our data in 2 sets :
- "training set"
- "test set" (or "hold-out set")

scikit-learn.train_test_split does it for you.
"""

from sklearn.model_selection import train_test_split

# It first shuffles the data set, 
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

# STEP 1 : visualize your data to detect anomalies 

"""
"scatter-plot" : one feature per axis
"pair-plot" : analyze the data by pair of features
"""
import pandas as pd
import mglearn

# create dataframe fom data in X_train
# labels the columns with string is iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter-matrix from the data_frame, color by y_train
# grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=8, cmap=mglearn.cm3)

"""
Keys of iris_dataset: 
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
.. _iris_dataset:

Iris plants dataset
--------------------

**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
                
    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988

The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
from Fisher's paper. Note that it's the same as in R, but not as in the UCI
Machine Learning Repository, which has two wrong data points.

This is perhaps the best known database to be found in the
pattern recognition literature.  Fisher's paper is a classic in the field and
is referenced frequently to this day.  (See Duda & Hart, for example.)  The
data set contains 3 classes of 50 instances each, where each class refers to a
type of iris plant.  One class is linearly separable from the other 2; the
latter are NOT linearly separable from each other.

.. topic:: References

   - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
     Mathematical Statistics" (John Wiley, NY, 1950).
   - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
     Structure and Classification Rule for Recognition in Partially Exposed
     Environments".  IEEE Transactions on Pattern Analysis and Machine
     Intelligence, Vol. PAMI-2, No. 1, 67-71.
   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
     on Information Theory, May 1972, 431-433.
   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
     conceptual clustering system finds 3 classes in the data.
   - Many, many more ...
...
Target names : ['setosa' 'versicolor' 'virginica']
Feature names : ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Type of data : <class 'numpy.ndarray'>
Shape of data : (150, 4)
First 5 columns of data : 
 [[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
Type of flowers measured : 
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
X_train shape: (112, 4)
y_train shape: (112,)
X_test shape: (38, 4)
y_test shape: (38,)


Chart output at pair-plot_1.png
"""


# Build the first model : K-Nearest Neighbors

"""
For this approach, we will store our training set, and for a new input,
we find the k-nearest neighbors for this new point, and the majority of 
their labels, will be our new point label.
"""

# SINGLE FLOWER TEST

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

import numpy as np

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction : {}".format(prediction))
print("Predicted target name : {}".format(iris_dataset['target_names'][prediction]))

# Evaluate accuracy of the model on the test-set

y_pred = knn.predict(X_test)
print("Test set predictions:\m {}".format(y_pred))

# y_pred == y_test will build an array of booleans.
# true if the values are equals, else otherwise.
# in python, true is 1.0, and false is 0.0
# so the mean gives you the accuracy score
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
# OR
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


"""
Code summary :

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

"""