#!/usr/bin/env python
from time import time
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
    return 100*np.mean(scores)

def SFS(data_train, target_train, classifier):
    rowsize = len(data_train[0])
    data_number = data_train.shape[0]
    selected_features = np.zeros(rowsize, dtype=np.bool)
    best_score = 0
    best_feature = 0
    scores = np.zeros(data_number, dtype=np.float32)

    while best_feature is not None:
        best_feature = None

        available_features = np.where(selected_features == False)

        for data_idx in available_features[0]:

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

    initial_sol = np.random.choice([True, False], rowsize)

    scores = np.zeros(data_number, dtype=np.float32)

    selected_features = initial_sol

    end = False

    best_score = score_solution(data_train, target_train, selected_features, scores, classifier)
    evaluations = 0
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
    initial_sol = np.random.choice([True, False], rowsize)
    selected_features = initial_sol
    scores = np.zeros(data_number, dtype=np.float32)

    best_score = score_solution(data_train, target_train, selected_features, scores, classifier)
    actual_score = best_score
    best_solution = np.copy(initial_sol)

    T0 = mu * best_score / (-np.log(phi))
    Tf = 1e-3

    max_neighbors = 10 * rowsize
    max_accepted = 0.1 * max_neighbors
    M = np.ceil(MAX_EVALUATIONS / max_neighbors)
    beta = (T0 - Tf) / (M * T0 * Tf)

    T = T0
    evaluations = 0
    accepted_neighbors = 1
    max_eval = MAX_EVALUATIONS

    # max_eval = 5000

    while T >= Tf and evaluations < max_eval and accepted_neighbors != 0:
        accepted_neighbors = 0
        for actual_neighbors in range(max_neighbors):
            if accepted_neighbors >= max_accepted:
                break
            feature = np.random.randint(rowsize)
            flip(selected_features, feature)
            new_score = score_solution(data_train, target_train, selected_features, scores, classifier)

            deltaF = actual_score - new_score

            evaluations += 1

            unif = np.random.uniform()
            coc = -deltaF/T
            ex = np.exp(coc)
            # print("deltaF=",deltaF, "T=",T,"coc=",coc, "unif=",unif, "expo=",ex, "delta!=0 = ",deltaF != 0)

            if (deltaF != 0) and (deltaF < 0 or unif < ex ):
                accepted_neighbors += 1
                actual_score = new_score
                if actual_score > best_score:
                    best_score = new_score
                    np.copyto(best_solution, selected_features)
            else:
                flip(selected_features, feature)

        T = T / (1 + beta * T)
        # print("T =",T, "eval =", evaluations, "tiempo =", time())
        # print("Best_sol =", best_score, "accepted_neighbors =", accepted_neighbors)

    return best_solution, best_score

def TS(data_train, target_train, classifier):
    rowsize = len(data_train[0])
    data_number = data_train.shape[0]

    initial_sol = np.random.choice([True, False], rowsize)

    scores = np.zeros(data_number, dtype=np.float32)

    selected_features = initial_sol

    best_score = score_solution(data_train, target_train, selected_features, scores, classifier)

    size_tabu_list = rowsize // 3
    tabu_list = np.repeat(-1,size_tabu_list)

    pos_tabu_list = 0

    evaluations = 0
    size_neighbourhood = 30

    while evaluations < MAX_EVALUATIONS:
        actual_score = 0
        best_feature = -1

        neighbourhood = np.random.choice(rowsize, size_neighbourhood, replace = False)

        for idx in neighbourhood:
            flip(selected_features, idx)
            # Check if there isn't any feature
            while sum(selected_features) == 0:
                feature = np.random.randint(rowsize)
                flip(selected_features, feature)

            new_score = score_solution(data_train, target_train, selected_features, scores, classifier)
            flip(selected_features, idx)

            evaluations += 1

            if idx in tabu_list:
                if new_score > best_score and new_score > actual_score:
                    actual_score = new_score
                    best_feature = idx
            elif new_score > actual_score:
                actual_score = new_score
                best_feature = idx

            if evaluations >= MAX_EVALUATIONS:
                break

        pos_tabu_list = (pos_tabu_list + 1) % size_tabu_list
        tabu_list[pos_tabu_list] = best_feature

        if (actual_score > best_score):
            best_score = actual_score
            flip(selected_features, best_feature)

    return selected_features, best_score

def TSext(data_train, target_train, classifier):
    rowsize = len(data_train[0])
    data_number = data_train.shape[0]

    initial_sol = np.random.choice([True, False], rowsize)

    scores = np.zeros(data_number, dtype=np.float32)
    frec = np.zeros(rowsize, dtype=np.int32)

    selected_features = initial_sol

    best_score = score_solution(data_train, target_train, selected_features, scores, classifier)

    size_tabu_list = rowsize // 3
    tabu_list = np.repeat(-1,size_tabu_list)

    pos_tabu_list = 0

    evaluations = 0
    size_neighbourhood = 30

    iterations_without_improve = 0

    best_solution = np.zeros(rowsize, dtype=np.bool)

    while evaluations < MAX_EVALUATIONS:
        actual_score = 0
        best_feature = -1

        if iterations_without_improve >= 10:
            # print("Reset!")
            iterations_without_improve = 0
            choice = np.random.uniform()
            if choice < 0.25: # New random solution
                selected_features = np.random.choice([True, False], rowsize) #initial solution
            elif choice < 0.5: # New from best solution
                selected_features = best_solution
            else: # Use memory
                total_sols = sum(frec)
                selected_features = np.array([(np.random.uniform() < 1 - i / total_sols) for i in frec], dtype=np.bool)

            change_size = np.random.uniform()
            if change_size < 0.5:
                size_tabu_list *= 1.5
            else:
                size_tabu_list /= 1.5
            size_tabu_list = np.ceil(size_tabu_list)
            tabu_list = np.zeros(size_tabu_list, dtype=np.int32)

        neighbourhood = np.random.choice(rowsize, size_neighbourhood, replace = False)

        for idx in neighbourhood:
            flip(selected_features, idx)
            # Check if there isn't any feature
            while sum(selected_features) == 0:
                feature = np.random.randint(rowsize)
                flip(selected_features, feature)

            new_score = score_solution(data_train, target_train, selected_features, scores, classifier)
            flip(selected_features, idx)

            evaluations += 1

            if idx in tabu_list:
                if new_score > best_score and new_score > actual_score:
                    actual_score = new_score
                    best_feature = idx
                    frec[idx] += 1
                else:
                    iterations_without_improve += 1
            elif new_score > actual_score:
                actual_score = new_score
                best_feature = idx
                frec[idx] += 1
            else:
                iterations_without_improve += 1

            if evaluations >= MAX_EVALUATIONS:
                break


        pos_tabu_list = (pos_tabu_list + 1) % size_tabu_list
        tabu_list[pos_tabu_list] = best_feature

        if (actual_score > best_score):
            best_score = actual_score
            flip(selected_features, best_feature)
            best_solution = np.copy(selected_features)

    return best_solution, best_score
