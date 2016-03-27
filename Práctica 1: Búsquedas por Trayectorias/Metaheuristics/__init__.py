#!/usr/bin/env python

import numpy as np
from sklearn import neighbors
from sklearn import cross_validation

def flip(selected_features, idx):
    selected_features[idx] = not selected_features[idx]

def SFS(data_train, target_train, classifier):
    rowsize = len(data_train[0])
    data_number = data_train.shape[0]
    selected_features = np.zeros(rowsize, dtype=np.bool)
    best_score = 0
    best_feature = 0
    scores = np.zeros(data_number, dtype=np.float32)

    while best_feature is not None:
        end = True
        best_feature = None

        for data_idx in range(rowsize):
            if selected_features[data_idx]:
                continue

            selected_features[data_idx] = True
            loo = cross_validation.LeaveOneOut(data_number)

            for idx, partition_idx in enumerate(loo):
                train_index = partition_idx[0]
                test_index = partition_idx[1]
                X_train, X_test = data_train[train_index], data_train[test_index]
                y_train, y_test = target_train[train_index], target_train[test_index]
                classifier.fit(X_train[:,selected_features], y_train)
                scores[idx] = classifier.score(X_test[:,selected_features], y_test)
            selected_features[data_idx] = False

            score = np.mean(scores)
            if score > best_score:
                best_score = score
                best_feature = data_idx

        if best_feature is not None:
            selected_features[best_feature] = True

    return selected_features, best_score

def LS(data_train, data_test, target_train, target_test, classifier, initial_sol = None):
    rowsize = len(data_train[0])
    if initial_sol is None:
        initial_sol = np.zeros(rowsize, dtype=np.bool)
        initial_sol[0] = True

    selected_features = initial_sol

    end = False

    classifier.fit(data_train[:,selected_features], target_train)
    best_score = classifier.score(data_test[:,selected_features], target_test)

    while not end:
        for idx, feature in enumerate(selected_features):
            if feature:
                continue

            flip(selected_features, idx)

            classifier.fit(data_train[:,selected_features], target_train)
            score = classifier.score(data_test[:,selected_features], target_test)

            flip(selected_features, idx)

            if score > best_score:
                best_score = score
                flip(selected_features, idx)
                break
        end = True

    return selected_features, best_score
