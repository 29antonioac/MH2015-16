#!/usr/bin/env python

import numpy as np
from sklearn import neighbors

def flip(selected_features, idx):
    selected_features[idx] = not selected_features[idx]

def SFS(data_train, data_test, target_train, target_test, classifier):
    rowsize = len(data_train[0])
    selected_features = np.zeros(rowsize, dtype=np.bool)
    best_score = 0
    best_feature = 0

    while best_feature is not None:
        end = True
        best_feature = None

        for idx in range(rowsize):
            if selected_features[idx]:
                continue

            selected_features[idx] = True
            classifier.fit(data_train[:,selected_features], target_train)
            score = classifier.score(data_test[:,selected_features], target_test)
            selected_features[idx] = False

            if score > best_score:
                best_score = score
                best_feature = idx

        if best_feature is not None:
            selected_features[best_feature] = True

    return selected_features, best_score

def LS(data_train, data_test, target_train, target_test, classifier):
    rowsize = len(data_train[0])
    selected_features = np.zeros(rowsize, dtype=np.bool)

    selected_features[0] = True

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
