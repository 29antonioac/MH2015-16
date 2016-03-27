#!/usr/bin/env python

import sys
import logging
import numpy as np
from scipy.io import arff
from sklearn import neighbors
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from time import time
from Metaheuristics import *

def main():
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    dataset, metadata = arff.loadarff("Data/wdbc.arff")
    # dataset, metadata = arff.loadarff("Data/movement_libras.arff")
    # dataset, metadata = arff.loadarff("Data/arrhythmia.arff")

    feature_names = metadata.names()[:-1]  # Everything but the last column
    data = dataset[feature_names]
    data = np.asarray(data.tolist(), dtype=np.float32) # Fix the np.void type

    m = MinMaxScaler()
    data = m.fit_transform(data)            # Normalize data

    target = dataset[:]["class"]            # Only the class column
    target = np.asarray(target.tolist(), dtype=np.int16)

    knn = neighbors.KNeighborsClassifier(n_neighbors = 3, n_jobs = -1)

    for iteration in range(5):
        skf = StratifiedKFold(target, n_folds=2, shuffle=True)

        for train_index, test_index in skf:
           data_train, data_test = data[train_index], data[test_index]
           target_train, target_test = target[train_index], target[test_index]

           start = time()
           selected_features, score = SFS(data_train, data_test, target_train, target_test, knn)
           end = time()

           logger.info("Time elapsed: " + str(end-start))

if __name__ == "__main__":
    main()
