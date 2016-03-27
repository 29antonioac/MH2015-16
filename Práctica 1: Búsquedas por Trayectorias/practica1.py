#!/usr/bin/env python

import sys
import logging
import numpy as np
from scipy.io import arff
from sklearn import datasets
from sklearn import neighbors
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import check_cv
from sklearn.preprocessing import MinMaxScaler

def main():
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    dataset, metadata = arff.loadarff("Datos/wdbc.arff")
    # dataset, metadata = arff.loadarff("Datos/movement_libras.arff")
    # dataset, metadata = arff.loadarff("Datos/arrhythmia.arff")

    data = dataset[metadata.names()[:-1]]   # Everything but the last column
    data = np.asarray(data.tolist(), dtype=np.float32) # Fix the np.void type
    m = MinMaxScaler()
    data = m.fit_transform(data)            # Normalize data

    target = dataset[:]["class"]            # Only the class column
    target = np.asarray(target.tolist(), dtype=np.int16)

    knn = neighbors.KNeighborsClassifier(n_neighbors = 3, n_jobs = -1)

    skf = StratifiedKFold(target, n_folds=2, shuffle=True)

    for train_index, test_index in skf:
       logger.debug("TRAIN: " + str(train_index) + " TEST: " + str(test_index))
       data_train, data_test = data[train_index], data[test_index]
       target_train, target_test = target[train_index], target[test_index]

       logger.debug("X_train: " + str(data_train) + " X_test: " + str(data_test))
       logger.debug("y_train: " + str(target_train) + " y_test: " + str(target_test))
       knn.fit(data_train, target_train)
       logger.info(knn.score(data_test,target_test))
    # knn.fit(iris.data, iris.target)

if __name__ == "__main__":
    main()
