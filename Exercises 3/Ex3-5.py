# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:13:28 2019

@author: Asier
"""

import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

mat = scipy.io.loadmat('twoCLassData.mat')

X = mat['X']
y = mat['y']
y = y.transpose()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

y_train = y_train.ravel()
y_test = y_train.ravel()


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print(neigh.score(X_test, y_test))

clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))