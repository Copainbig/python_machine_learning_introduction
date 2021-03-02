# NUMPY
# NumPy offers a collection of high level maths functions and functionality for multi-dimensional arrays

import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))

"""
x:
[[1 2 3]
 [4 5 6]]
 """

# SCIPY
# SciPy offers a collection of scientific computing functions

from scipy import sparse

eye = np.eye(4)
print("Numpy array:\{}", format(eye))

"""
Numpy array:\{} [[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
"""

# Convert the Numpy array into a SciPy sparse Matrix, in CSR format
# That only stores the non-zero entries
sparse_matrix = sparse.csr_matrix(eye)
print("\n SciPy sparse CSR matrix:\n{}".format(sparse_matrix))

"""
 SciPy sparse CSR matrix:
  (0, 0)        1.0
  (1, 1)        1.0
  (2, 2)        1.0
  (3, 3)        1.0
"""

# used when Dense representation to not fit in memory
# COOrdinate representation is lighter for sparse matrix
data = np.ones(4)           # [1., 1., 1., 1.]
row_indices = np.arange(4)  # [0, 1 ,2 ,3]
col_indices = np.arange(4)  # [0, 1, 2, 3]
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n{}".format(eye_coo))

"""
COO representation:
  (0, 0)        1.0
  (1, 1)        1.0
  (2, 2)        1.0
  (3, 3)        1.0
"""

# MATPLOTLIB
# Ploting library for visualization

import matplotlib.pyplot as plt

# generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# create a second array using sine
y = np.sin(x)
# the plot function makes a line chart of one array against another
plt.plot(x, y, marker='x')
plt.show()
# outputs matplotlib_example1.png

# PANDAS
# library used to analyse data, based on a table like data-structure called data-frame
# it can also be used to ingest different data types (CSV...)

import pandas as pd

# create a simple dataset of people
data = {
  'name': ["John", "Anna", "Peter", "Linda"],
  'Location': ["New York", "Paris", "Berlin", "London"],
  'Age': [24, 13, 53, 33]
}

data_pandas = pd.DataFrame(data)
# IPython.display allows "pretty printing" of dataframes, in the Jupyter Notebook
# display(data_pandas)