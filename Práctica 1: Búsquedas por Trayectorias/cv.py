#!/usr/bin/env python

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import check_cv
from sklearn.preprocessing import MinMaxScaler

np.random.seed(123456789)

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 0, 1, 1])

print(X)
print(y)
skf = StratifiedKFold(y, n_folds=2, shuffle=True)
len(skf)

print(skf)


for train_index, test_index in skf:
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   print("X_train:", X_train, "X_test:", X_test)
   print("y_train:", y_train, "y_test:", y_test)

m = MinMaxScaler()
X = m.fit_transform(X)
print(X)
