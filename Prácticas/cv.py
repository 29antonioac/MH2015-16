#!/usr/bin/env python

import numpy as np
from sklearn import neighbors
from time import time
from sklearn import datasets
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('forkserver', force = True)
    np.random.seed(123456)

    iris = datasets.load_iris()

    knn = neighbors.KNeighborsClassifier(n_neighbors = 3, n_jobs = 2)

    perm = np.random.permutation(iris.target.size)
    iris.data = iris.data[perm]
    iris.target = iris.target[perm]
    knn.fit(iris.data[:100], iris.target[:100])

    start = time()
    score = knn.score(iris.data[100:], iris.target[100:])
    end = time()

    print(score)
    print(end-start)
