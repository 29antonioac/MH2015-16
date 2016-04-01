#!/usr/bin/env python

import numpy as np
from sklearn import neighbors
from sklearn import cross_validation

MAX_EVALUATIONS = 15000

def flip(selected_features, idx):
    selected_features[idx] = not selected_features[idx]

def score_solution(data_train, target_train, selected_features, scores, classifier):
    data_number = data_train.shape[0]
    loo = cross_validation.LeaveOneOut(data_number)

    for idx, partition_idx in enumerate(loo):
        train_index = partition_idx[0]
        test_index = partition_idx[1]
        X_train, X_test = data_train[train_index], data_train[test_index]
        y_train, y_test = target_train[train_index], target_train[test_index]
        classifier.fit(X_train[:, selected_features], y_train)
        scores[idx] = classifier.score(X_test[:, selected_features], y_test)
    return np.mean(scores)

def SFS(data_train, target_train, classifier):
    rowsize = len(data_train[0])
    data_number = data_train.shape[0]
    selected_features = np.zeros(rowsize, dtype=np.bool)
    best_score = 0
    best_feature = 0
    scores = np.zeros(data_number, dtype=np.float32)

    while best_feature is not None:
        best_feature = None

        for data_idx in range(rowsize):
            if selected_features[data_idx]:
                continue

            selected_features[data_idx] = True
            score = score_solution(data_train, target_train, selected_features, scores, classifier)
            selected_features[data_idx] = False

            if score > best_score:
                best_score = score
                best_feature = data_idx

        if best_feature is not None:
            selected_features[best_feature] = True

    return selected_features, best_score

def LS(data_train, target_train, classifier):
    rowsize = len(data_train[0])
    data_number = data_train.shape[0]

    initial_sol = np.zeros(rowsize, dtype=np.bool)
    initial_sol[np.random.randint(2)] = True

    scores = np.zeros(data_number, dtype=np.float32)

    selected_features = initial_sol

    end = False

    best_score = score_solution(data_train, target_train, selected_features, scores, classifier)
    evaluations = 1
    while not end and evaluations < MAX_EVALUATIONS:
        l_neighbors = list(enumerate(selected_features))
        np.random.shuffle(l_neighbors)
        for idx, feature in l_neighbors:
            if feature:
                continue
            if evaluations >= MAX_EVALUATIONS:
                break

            flip(selected_features, idx)

            score = score_solution(data_train, target_train, selected_features, scores, classifier)

            flip(selected_features, idx)

            if score > best_score:
                best_score = score
                flip(selected_features, idx)
                break
        else:
            end = True

    return selected_features, best_score

def SA(data_train, target_train, classifier):
    mu = 0.3
    phi = 0.3
    rowsize = len(data_train[0])
    data_number = data_train.shape[0]
    initial_sol = np.zeros(rowsize, dtype=np.bool)
    initial_sol[np.random.randint(2)] = True
    selected_features = initial_sol
    scores = np.zeros(data_number, dtype=np.float32)

    best_score = score_solution(data_train, target_train, selected_features, scores, classifier)
    actual_score = best_score

    T0 = mu * best_score / (-np.log(phi))
    Tf = 1e-3
    
    max_neighbors = 10 * rowsize
    max_accepted = 0.1 * max_neighbors
    M = np.ceil(MAX_EVALUATIONS / max_neighbors)

    T = T0
    evaluations = 0
    actual_neighbors = 0
    accepted_neighbors = 0

    while T >= Tf and evaluations < MAX_EVALUATIONS:
        while evaluations < MAX_EVALUATIONS and actual_neighbors < max_neighbors and accepted_neighbors < max_accepted:
            feature = np.random.randint(rowsize)
            flip(selected_features, feature)
            new_score = score_solution(data_train, target_train, selected_features, scores, classifier)
            flip(selected_features, feature)
            deltaF = new_score - actual_score

            evaluations += 1
            actual_neighbors += 1
            if (deltaF != 0) and (deltaF > 0 or np.random.uniform() < np.exp(deltaF/T)):
                accepted_neighbors += 1
                actual_score = new_score
                if actual_score > best_score:
                    best_score = new_score
                    flip(selected_features, feature)

        beta = (T0 - Tf) / (M * T0 * Tf)
        T = T / (1 + beta * T)

    return selected_features, best_score
