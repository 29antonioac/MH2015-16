#!/usr/bin/env python

import numpy as np
from scipy.io import arff
from sklearn import datasets
from sklearn import neighbors
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import check_cv
from sklearn.preprocessing import MinMaxScaler

dataset, metadata = arff.loadarff("Datos/wdbc.arff")

data = dataset[metadata.names()[:-1]]   # Everything but the last column
data = np.asarray(data.tolist(), dtype=np.float32) # Fix the np.void type

target = dataset[:]["class"]            # Only the class column
target = np.asarray(target.tolist(), dtype=np.int16)

knn = neighbors.KNeighborsClassifier(n_neighbors = 3, n_jobs = -1)

knn.fit(iris.data, iris.target)
