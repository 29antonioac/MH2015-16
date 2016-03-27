#!/usr/bin/env python

import numpy as np
from sklearn import neighbors

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
