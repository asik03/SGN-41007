# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:52:14 2019

@author: Asier
"""
import traffic_signs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

C_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

features = []
X, y = traffic_signs.load_data('./data')

#### Load or  Extract features
#features = traffic_signs.extract_lbp_features(X)
features = np.load('./data/features.npy')


#features = np.array(features)    
#features = features.reshape(features.shape[0], features.shape[1]*features.shape[2])
#X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

## Spliting data
X_train, X_test, y_train, y_test = train_test_split(features, y)


parameters = {'kernel':('linear', 'rbf'), 'C':C_range}
svc = SVC(gamma="scale")

clf1 = GridSearchCV(svc, parameters, cv=5)

# In contrast to GridSearchCV, not all parameter values are tried out, 
# but rather a fixed number of parameter settings is sampled from the 
# specified distributions. The number of parameter settings that are tried 
# is given by n_iter.
clf2 = RandomizedSearchCV(svc, parameters, cv=5, n_iter = 15)

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)