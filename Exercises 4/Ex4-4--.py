# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:24:59 2019

@author: Asier
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.image as mpimg
from math import floor, ceil
import os

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import glob

import matplotlib.pyplot as plt
from simplelbp import local_binary_pattern

def load_data(folder):
    """ 
    Load all images from subdirectories of
    'folder'. The subdirectory name indicates
    the class.
    """
    
    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here
    
    subdirectories = glob.glob(folder + "/*")
    
    # Loop over all folders
    for d in subdirectories:
        
        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")
        
        # Load all files
        for name in files:
            
            # Load image and parse class name
            img = plt.imread(name)
            class_name = name.split(os.sep)[-2]

            # Convert class names to integer indices:
            if class_name not in classes:
                classes.append(class_name)
            
            class_idx = classes.index(class_name)
            
            X.append(img)
            y.append(class_idx)
    
    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def extract_lbp_features(X, P = 8, R = 5):
    """
    Extract LBP features from all input samples.
    - R is radius parameter
    - P is the number of angles for LBP
    """
    
    F = [] # Features are stored here
    
    N = X.shape[0]
    for k in range(N):
        
        print("Processing image {}/{}".format(k+1, N))
        
        image = X[k, ...]
        lbp = local_binary_pattern(image, P, R)
        hist = np.histogram(lbp, bins=range(257))[0]
        F.append(hist)

    return np.array(F)

# Test our loader

X, y = load_data(".")
F = extract_lbp_features(X)
print("X shape: " + str(X.shape))
print("F shape: " + str(F.shape))
images = []
labels = []
features = []
scores = []

class_1_path = './GTSRB_subset/class1/'
class_2_path = './GTSRB_subset/class2/'    


# Load the images
for filepath in os.listdir(class_1_path):
    images.append(mpimg.imread(class_1_path + filepath))
    labels.append(0)

for filepath in os.listdir(class_1_path):
    images.append(mpimg.imread(class_1_path + filepath))
    labels.append(1)

# Extract features
for image in images:
    features.append(local_binary_pattern(image))

features = np.array(features)    
features = features.reshape(features.shape[0], features.shape[1]*features.shape[2])

X_train, X_test, y_train, y_test = train_test_split(features, labels)


## Training with different classifiers
# Nearest neighbor
neigh = KNeighborsClassifier(n_neighbors=3)
scores_neigh = cross_val_score(neigh, features, labels, cv=5)
print("Neigh score:")
print(scores_neigh)
neigh.fit(X_train, y_train)
neigh_score = accuracy_score(y_test, neigh.predict(X_test))
scores.append(neigh_score)

# LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
scores_lda = cross_val_score(lda, features, labels, cv=5)
print("Lda score: ") 
print(scores_lda)
lda.fit(X_train, y_train)
lda_score = accuracy_score(y_test, lda.predict(X_test))
scores.append(lda_score)

# SVM
svm = SVC(gamma='auto')
scores_svm = cross_val_score(svm, features, labels, cv=5)
print("Svm score:")
print( scores_svm)
svm.fit(X_train, y_train)
svm_score = accuracy_score(y_test, svm.predict(X_test))
scores.append(svm_score)

# Logistic Regrasion
lmr = LogisticRegression()
scores_lmr = cross_val_score(lmr, features, labels, cv=5)
print("Lmr score:")
print(scores_lmr)
lmr.fit(X_train, y_train)
lmr_score = accuracy_score(y_test, lmr.predict(X_test))
scores.append(lmr_score)

for i in scores:
    print(i)
# Values are different every testing, not using a random state
