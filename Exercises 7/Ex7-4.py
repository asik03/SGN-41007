# -*- coding: utf-8 -*-
import scipy.io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

mat = scipy.io.loadmat("./input/arcene.mat")
y_train = np.ravel(mat["y_train"])
y_test = np.ravel(mat["y_test"])
X_train = mat["X_train"]
X_test = mat["X_test"]

est = LogisticRegression(solver="lbfgs")

rfe = RFECV(estimator=est, step=50,verbose=1)
rfe = rfe.fit(X_train, y_train)

rfe.support_

plt.plot(range(0,10001,50), rfe.grid_scores_)

score = accuracy_score(y_test, rfe.predict(X_test))

print('Test accuracy', score)
