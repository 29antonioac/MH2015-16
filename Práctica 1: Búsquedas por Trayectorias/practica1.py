#!/usr/bin/env python

import numpy as np
from sklearn import datasets
from sklearn import neighbors

iris = datasets.load_iris()
knn = neighbors.KNeighborsClassifier(n_neighbors = 3, n_jobs = -1)
knn.fit(iris.data, iris.target)
prediction = knn.predict([[0.1, 0.2, 0.3, 0.4]])

print(prediction)
