# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:52:14 2019

@author: Asier
"""
import traffic_signs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

C_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
features = []
scores = []
X, y = traffic_signs.load_data('./data')

#### Load or  Extract features
#features = traffic_signs.extract_lbp_features(X)
features = np.load('./data/features.npy')

#features = np.array(features)    
#features = features.reshape(features.shape[0], features.shape[1]*features.shape[2])
#X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

## Spliting data
X_train, X_test, y_train, y_test = train_test_split(features, y)

clf_list = [LogisticRegression(), SVC(gamma='scale')]
clf_name = ['LR', 'SVC']
for clf,name in zip(clf_list, clf_name):
    for C in C_range:
        for penalty in ["l1", "l2"]:
                clf.C = C
                clf.penalty = penalty
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                scores.append(score)
                print(name + " cls with penalty '" + penalty + " and C: " + str(C) + "'. Score: " + str(score))