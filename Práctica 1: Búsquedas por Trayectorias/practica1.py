#!/usr/bin/env python


import sys
import logging

import pystache
import numpy as np
from scipy.io import arff
from sklearn import neighbors
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from collections import OrderedDict
from time import time
from Metaheuristics import *


def main():

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    algorithms_table = {}
    algorithms = [SFS,LS,SA,TS]

    databases = {   "W" : "Data/wdbc.arff",
                    "M" : "Data/movement_libras.arff",
                    "A" : "Data/arrhythmia.arff"}

    databases = OrderedDict(sorted(databases.items(), key=lambda t: t[0]))
    databases = OrderedDict(reversed(databases.items()))

    print(databases)

    # databases = { "W" : "Data/wdbc.arff" }

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
        repeats = 5
        n_folds = 2

        for iteration in range(repeats):
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

    W_clas_in = 0
    W_clas_out = 0
    W_red = 0
    W_T = 0

    M_clas_in = 0
    M_clas_out = 0
    M_red = 0
    M_T = 0

    A_clas_in = 0
    A_clas_out = 0
    A_red = 0
    A_T = 0


    for alg in algorithms:
        for it in algorithms_table[alg.__name__]["items"]:
            W_clas_in += it["W_clas_in"]
            W_clas_out += it["W_clas_out"]
            W_red += it["W_red"]
            W_T += it["W_T"]

            M_clas_in += it["M_clas_in"]
            M_clas_out += it["M_clas_out"]
            M_red += it["M_red"]
            M_T += it["M_T"]

            A_clas_in += it["A_clas_in"]
            A_clas_out += it["A_clas_out"]
            A_red += it["A_red"]
            A_T += it["A_T"]
        W_clas_in /= repeats*n_folds
        W_clas_out /= repeats*n_folds
        W_red /= repeats*n_folds
        W_T /= repeats*n_folds

        M_clas_in /= repeats*n_folds
        M_clas_out /= repeats*n_folds
        M_red /= repeats*n_folds
        M_T /= repeats*n_folds

        A_clas_in /= repeats*n_folds
        A_clas_out /= repeats*n_folds
        A_red /= repeats*n_folds
        A_T /= repeats*n_folds

        algorithms_table[alg.__name__]["items"].append({
            "name" : "Media",
            "W_clas_in" : W_clas_in,
            "W_clas_out" : W_clas_out,
            "W_red" : W_red,
            "W_T" : W_T,
            "M_clas_in" : M_clas_in,
            "M_clas_out" : M_clas_out,
            "M_red" : M_red,
            "M_T" : M_T,
            "A_clas_in" : A_clas_in,
            "A_clas_out" : A_clas_out,
            "A_red" : A_red,
            "A_T" : A_T
        })


    with open("Other/template.mu", "r") as template:
        for alg in algorithms:
            with open("Other/"+alg.__name__+"-result.mu", "w") as dest:
                dest.write(pystache.render(template.read(), algorithms_table[alg.__name__]))
if __name__ == "__main__":
    #import multiprocessing as mp; mp.set_start_method('forkserver', force = True)
    main()
