# -*- coding: utf-8 -*-
import scipy.io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
#from sklearn.model_selection import GridSearchCV

C_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
scores = []
parameters = {'C':C_range}

mat = scipy.io.loadmat("./input/arcene.mat")
y_train = np.ravel(mat["y_train"])
y_test = np.ravel(mat["y_test"])
X_train = mat["X_train"]
X_test = mat["X_test"]

model = LogisticRegression(penalty='l1')

for C in C_range:
    model.C = C
    model.penalty = 'l1'
    model.fit(X_train, y_train)
    model.coef_
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
    print("Cls with penalty l1 and C: " + str(C) + "'. Score: " + str(score))