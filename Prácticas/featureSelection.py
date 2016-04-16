#!/usr/bin/env python


import sys
import logging
import random

import pystache
import numpy as np
import multiprocessing
from scipy.io import arff
from sklearn import neighbors
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from collections import OrderedDict
from time import time
from time import sleep
from Metaheuristics import *

from Metaheuristics.knnGPU.knnLooGPU import knnLooGPU



def main(algorithm):

    print("Launching",algorithm)

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    arg_algorithms = {"-KNN" : KNN, "-SFS" : SFS, "-LS" : LS, "-SA" : SA, "-TS" : TS, "-TSext" : TSext,
                        "-BMB" : BMB, "-GRASP" : GRASP, "-ILS" : ILS,
                        "-GA" : GA }
    algorithm_table = {}
    alg = arg_algorithms[algorithm]

    algorithm_table["algorithm"] = alg.__name__

    databases = OrderedDict([('W', 'Data/wdbc.arff'), ('M', 'Data/movement_libras.arff'), ('A', 'Data/arrhythmia.arff')])

    np.set_printoptions(precision=5)

    for key, value in databases.items():
        np.random.seed(12345678)
        random.seed(12345678)

        dataset, metadata = arff.loadarff(value)

        feature_names = metadata.names()[:-1]  # Everything but the last column
        data = dataset[feature_names]
        data = np.asarray(data.tolist(), dtype=np.float32) # Fix the np.void type

        m = MinMaxScaler()
        data = m.fit_transform(data)            # Normalize data

        target = dataset[:]["class"]            # Only the class column
        target = np.asarray(target.tolist(), dtype=np.int32)

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
                scorerGPU = knnLooGPU(data_train.shape[0], data_train.shape[1], 3)

                actual_items = algorithm_table.setdefault("items", [{} for _ in range(repeats*n_folds)])
                item = actual_items[2*iteration + run]


                start = time()
                selected_features, score = alg(data_train, target_train, scorerGPU)
                end = time()

                knn.fit(data_train[:,selected_features], target_train)
                score_out = 100*knn.score(data_test[:,selected_features], target_test)

                item["name"] = "Partición " + str(iteration+1) + "-" + str(run+1)
                item[key+"_clas_in"] = float("{:,.5f}".format(score))
                item[key+"_clas_out"] = float("{:,.5f}".format(score_out))
                item[key+"_red"] = float("{:,.5f}".format( 100*(len(selected_features) - sum(selected_features)) / len(selected_features) ))
                item[key+"_T"] = float("{:,.5f}".format(end-start).replace(',',''))

                logger.info(key + " - " + str(alg.__name__) + " - Time elapsed: " + str(end-start) + ". Score: " + str(score) + ". Score out: " + str(score_out) + " Selected features: " + str(sum(selected_features)))
                # sleep(5)
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


    for it in algorithm_table["items"]:
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

    algorithm_table["items"].append({
        "name" : "Media",
        "W_clas_in" : "{:,.5f}".format(W_clas_in),
        "W_clas_out" : "{:,.5f}".format(W_clas_out),
        "W_red" : "{:,.5f}".format(W_red),
        "W_T" : "{:,.5f}".format(W_T).replace(',',''),
        "M_clas_in" : "{:,.5f}".format(M_clas_in),
        "M_clas_out" : "{:,.5f}".format(M_clas_out),
        "M_red" : "{:,.5f}".format(M_red),
        "M_T" : "{:,.5f}".format(M_T).replace(',',''),
        "A_clas_in" : "{:,.5f}".format(A_clas_in),
        "A_clas_out" : "{:,.5f}".format(A_clas_out),
        "A_red" : "{:,.5f}".format(A_red),
        "A_T" : "{:,.5f}".format(A_T).replace(',','')
    })


    with open("Other/template.mu", "r") as template:
            with open("Informes/Tables/"+alg.__name__+"-result.tex", "w") as dest:
                dest.write(pystache.render(template.read(), algorithm_table))
if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) > 2:
        # p = multiprocessing.Pool(min(len(sys.argv) - 1,multiprocessing.cpu_count()))
        print("HOLI",sys.argv[1:])
        for input in sys.argv[1:]:
            main(input)
    else:
        print("Give me an algorithm")
        print("-KNN, -SFS, -LS, -SA, -TS, -TSext")
