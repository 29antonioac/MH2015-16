#!/usr/bin/env python


import sys
import logging
import pystache
import numpy as np
from scipy.io import arff
from sklearn import neighbors
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from time import time
from Metaheuristics import *


def main():

    np.random.seed(12345678)

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    algorithms_table = {}
    algorithms = [LS]

    databases = {   "W" : "Data/wdbc.arff",
                    "M" : "Data/movement_libras.arff",
                    "A" : "Data/arrhythmia.arff"}

    for key, value in databases.items():
        np.random.seed(12345678)

        dataset, metadata = arff.loadarff(value)

        feature_names = metadata.names()[:-1]  # Everything but the last column
        data = dataset[feature_names]
        data = np.asarray(data.tolist(), dtype=np.float32) # Fix the np.void type

        m = MinMaxScaler()
        data = m.fit_transform(data)            # Normalize data

        target = dataset[:]["class"]            # Only the class column
        target = np.asarray(target.tolist(), dtype=np.int16)

        knn = neighbors.KNeighborsClassifier(n_neighbors = 3, n_jobs = 1)
        repeats = 1
        n_folds = 2

        for iteration in range(repeats):
            print("Iteration",iteration,"/",repeats)
            skf = StratifiedKFold(target, n_folds=n_folds, shuffle=True)

            for run, partition in enumerate(skf):
                train_index = partition[0]
                test_index = partition[1]
                data_train, data_test = data[train_index], data[test_index]
                target_train, target_test = target[train_index], target[test_index]

                for alg in algorithms:
                    actual_table = algorithms_table.setdefault(alg.__name__, {})
                    actual_items = actual_table.setdefault("items", [{} for _ in range(repeats*n_folds)])
                    item = actual_items[2*iteration + run]

                    start = time()
                    selected_features, score = alg(data_train, target_train, knn)
                    end = time()

                    knn.fit(data_train[:,selected_features], target_train)
                    score_out = knn.score(data_test[:,selected_features], target_test)

                    item["name"] = "Partición " + str(iteration+1) + "-" + str(run+1)
                    item[key+"_clas_in"] = score
                    item[key+"_clas_out"] = score_out
                    item[key+"_red"] = 100*(len(selected_features) - sum(selected_features)) / len(selected_features)
                    item[key+"_T"] = end-start

                    logger.info(key + " - " + str(alg.__name__) + " - Time elapsed: " + str(end-start) + ". Score: " + str(score) + ". Score out: " + str(score_out) + " Selected features: " + str(sum(selected_features)))
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    print("Volcando tablas...")
    pp.pprint(algorithms_table)
    print("-----------------")
    pp.pprint(algorithms_table["LS"])
    with open("Other/template.mu", "r") as template:
        for alg in algorithms:
            with open("Other/"+alg.__name__+"-result.mu", "w") as dest:
                dest.write(pystache.render(template.read(), algorithms_table[alg.__name__]))
if __name__ == "__main__":
    #import multiprocessing as mp; mp.set_start_method('forkserver', force = True)
    main()
